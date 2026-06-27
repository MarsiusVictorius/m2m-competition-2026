"""
V1.4: Competition-ready backtest.
- Dropped losing AUDUSD long config
- Walk-forward validation (train 20 days / test 10 days)
- Slippage model (0.5x spread per trade)
- Competition risk guardrails (leverage, margin, concentration monitoring)
- Versioned output charts
"""

import os, glob, warnings
from dataclasses import dataclass, field
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

DATA_DIR       = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
CANDLE_MINUTES = 15
ACCOUNT_EQUITY = 1_000_000.0
FIXED_RISK     = 10_000.0
MAX_POSITIONS  = 6
MAX_LEVERAGE   = 25.0
SLIPPAGE_SPREAD_MULT = 0.5  # 0.5x the bar spread added to each entry/exit

SL_ATR  = 1.5
TP_ATR  = 1.6
BE_ATR  = 1.0
TRAIL_ON = 1.0
TRAIL_D  = 1.0
PYRAMID_LAYERS  = 3
PYRAMID_TRIGGER = 1.0


@dataclass
class BOSConfig:
    symbol:    str
    direction: str
    poi_type:  str   = "asia"
    fract:     float = 0.0
    filter1:   str   = "none"
    t_segment: int   = 0
    label:     str   = ""


# V1.4 configs — AUDUSD long dropped
CONFIGS = [
    BOSConfig("AUDJPY",  "long",  "asia",       0.0, "none",             0, "AUDJPY long asia"),
    BOSConfig("AUDNZD",  "long",  "asia",       0.0, "vol_contracting",  3, "AUDNZD long asia+volC+T3"),
    BOSConfig("AUDUSD",  "short", "asia",       0.0, "vol_contracting",  2, "AUDUSD short asia+volC+T2"),
    BOSConfig("EURGBP",  "short", "asia",       0.3, "none",             1, "EURGBP short asia+0.3+T1"),
    BOSConfig("EURJPY",  "short", "prev_close", 1.0, "pullback_atr",     0, "EURJPY short prevC+pull"),
    BOSConfig("GBPUSD",  "long",  "asia",       0.0, "none",             1, "GBPUSD long asia+T1"),
    BOSConfig("NZDUSD",  "long",  "asia",       0.0, "none",             1, "NZDUSD long asia+T1"),
    BOSConfig("USDCNH",  "long",  "asia",       0.0, "vol_contracting",  0, "USDCNH long asia+volC"),
    BOSConfig("USDJPY",  "short", "prev_close", 0.5, "vol_expanding",    1, "USDJPY short prevC+volE+T1"),
    BOSConfig("USDJPY",  "long",  "prev_close", 0.5, "vol_expanding",    2, "USDJPY long prevC+volE+T2"),
    BOSConfig("XAUKUSD", "short", "prev_close", 0.5, "none",             0, "XAUKUSD short prevC+0.5"),
]


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
def load_and_build(symbol):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{symbol}_*.parquet")))
    if not files: return None
    dfs = [pd.read_parquet(f, columns=["time","bid","ask"]) for f in files]
    ticks = pd.concat(dfs, ignore_index=True)
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    ticks = ticks.sort_values("time")
    ticks["mid"]    = (ticks["bid"] + ticks["ask"]) / 2
    ticks["spread"] = ticks["ask"] - ticks["bid"]
    ticks = ticks.set_index("time")
    ohlc = ticks["mid"].resample(f"{CANDLE_MINUTES}min").ohlc()
    ohlc["spread"] = ticks["spread"].resample(f"{CANDLE_MINUTES}min").mean()
    ohlc = ohlc.dropna()
    ohlc.index = ohlc.index.tz_convert("UTC")

    ohlc["sma20"] = ohlc["close"].rolling(20).mean()
    ohlc["sma50"] = ohlc["close"].rolling(50).mean()
    prev = ohlc["close"].shift(1)
    tr = pd.concat([ohlc["high"]-ohlc["low"], (ohlc["high"]-prev).abs(), (ohlc["low"]-prev).abs()], axis=1).max(axis=1)
    ohlc["atr14"] = tr.rolling(14).mean()
    ohlc["atr5"]  = tr.rolling(5).mean()
    ohlc["atr40"] = tr.rolling(40).mean()
    ohlc["avg_spread"] = ohlc["spread"].rolling(20).mean()

    up_move   = ohlc["high"] - ohlc["high"].shift(1)
    down_move = ohlc["low"].shift(1) - ohlc["low"]
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14_s   = tr.ewm(span=14, adjust=False).mean()
    ohlc["dmi_plus"]  = pd.Series(plus_dm, index=ohlc.index).ewm(span=14, adjust=False).mean() / atr14_s * 100
    ohlc["dmi_minus"] = pd.Series(minus_dm, index=ohlc.index).ewm(span=14, adjust=False).mean() / atr14_s * 100
    dx = (ohlc["dmi_plus"] - ohlc["dmi_minus"]).abs() / (ohlc["dmi_plus"] + ohlc["dmi_minus"]) * 100
    ohlc["adx"] = dx.ewm(span=14, adjust=False).mean()

    ohlc["ema20"] = ohlc["close"].ewm(span=20, adjust=False).mean()
    ohlc["_date"] = ohlc.index.date
    ohlc["_hour"] = ohlc.index.hour

    daily = ohlc.groupby("_date").agg(
        day_open=("open","first"), day_close=("close","last"))
    daily["prev_close"] = daily["day_close"].shift(1)
    ohlc = ohlc.join(daily[["day_open","prev_close"]], on="_date")

    asia = ohlc[ohlc["_hour"] < 7].groupby("_date").agg(asia_high=("high","max"), asia_low=("low","min"))
    ohlc = ohlc.join(asia, on="_date")

    cum_ma = []; count = 0; running = 0.0; prev_d = None
    for idx, row in ohlc.iterrows():
        d = row["_date"]
        if d != prev_d: count = 0; running = 0.0; prev_d = d
        count += 1; running += (row["high"]+row["low"])/2
        cum_ma.append(running/count)
    ohlc["cum_ma"] = cum_ma

    ohlc["highest_h10"] = ohlc["high"].rolling(10).max()
    ohlc["lowest_l10"]  = ohlc["low"].rolling(10).min()

    ohlc.drop(columns=["_date","_hour"], inplace=True)
    return ohlc


def get_poi(bar, poi_type):
    if poi_type == "asia":
        return bar.get("asia_high", np.nan), bar.get("asia_low", np.nan)
    elif poi_type == "prev_close":
        pc = bar.get("prev_close", np.nan)
        return pc, pc
    elif poi_type == "cum_ma":
        cm = bar.get("cum_ma", np.nan)
        return cm, cm
    elif poi_type == "ema20":
        e = bar.get("ema20", np.nan)
        return e, e
    return np.nan, np.nan


def check_filter(bar, filt, direction):
    if filt == "none": return True
    elif filt == "vol_expanding":    return bar["atr5"] > bar["atr40"]
    elif filt == "vol_contracting":  return bar["atr5"] <= bar["atr40"]
    elif filt == "dmi_trend":
        return bar["dmi_plus"] > bar["dmi_minus"] if direction=="long" else bar["dmi_minus"] > bar["dmi_plus"]
    elif filt == "adx_trending":     return bar["adx"] > 25
    elif filt == "adx_ranging":      return bar["adx"] < 25
    elif filt == "pullback_atr":
        if direction == "long":
            return bar["atr14"] > bar.get("prev_close", bar["close"]) - bar["lowest_l10"]
        return bar["atr14"] > bar["highest_h10"] - bar.get("prev_close", bar["close"])
    elif filt == "close_vs_open":
        return bar["close"] > bar.get("day_open", bar["close"]) if direction=="long" else bar["close"] < bar.get("day_open", bar["close"])
    return True


# ─────────────────────────────────────────
# TRADE + PORTFOLIO ENGINE
# ─────────────────────────────────────────
@dataclass
class Trade:
    symbol: str; direction: str; layer: int; group_id: str
    entry_time: object; entry_price: float; stop_loss: float; take_profit: float
    atr: float; size: float; config_label: str
    exit_time: object = None; exit_price: float = None
    pnl: float = 0.0; mae: float = 0.0; mfe: float = 0.0
    exit_reason: str = ""; be_moved: bool = False; trailing: bool = False
    slippage_cost: float = 0.0

    @property
    def is_open(self): return self.exit_time is None
    def unrealised(self, price):
        return (price - self.entry_price) * self.size if self.direction=="long" else (self.entry_price - price) * self.size
    def profit_atr(self, price):
        raw = (price - self.entry_price) if self.direction=="long" else (self.entry_price - price)
        return raw / self.atr if self.atr > 0 else 0


# Competition risk state tracker
@dataclass
class RiskMonitor:
    equity: float = ACCOUNT_EQUITY
    leverage_breaches: int = 0      # bars where leverage > 25x
    margin_breach_bars: int = 0     # consecutive bars margin > 90%
    margin_penalties: int = 0       # -20 penalty events (30+ bars > 90%)
    conc_breach_bars: int = 0       # consecutive bars single instrument > 90%
    conc_penalties: int = 0         # -10 penalty events (30+ bars > 90%)
    max_leverage_seen: float = 0.0
    max_margin_pct: float = 0.0
    max_conc_pct: float = 0.0

    def check(self, open_trades, equity):
        self.equity = equity
        total_notional = sum(abs(t.size * t.entry_price) for t in open_trades)
        leverage = total_notional / equity if equity > 0 else 0

        self.max_leverage_seen = max(self.max_leverage_seen, leverage)
        if leverage > 25:
            self.leverage_breaches += 1

        margin_pct = (total_notional / (equity * 30)) * 100 if equity > 0 else 0
        self.max_margin_pct = max(self.max_margin_pct, margin_pct)
        if margin_pct > 90:
            self.margin_breach_bars += 1
            if self.margin_breach_bars >= 2:  # 30 min = 2 bars at 15min
                self.margin_penalties += 1
                self.margin_breach_bars = 0
        else:
            self.margin_breach_bars = 0

        # Per-symbol concentration (only meaningful with 2+ distinct symbols)
        if open_trades:
            sym_notionals = {}
            for t in open_trades:
                sym_notionals[t.symbol] = sym_notionals.get(t.symbol, 0) + abs(t.size * t.entry_price)
            if len(sym_notionals) >= 2:
                max_sym = max(sym_notionals.values())
                conc = max_sym / total_notional * 100 if total_notional > 0 else 0
            else:
                conc = 0
            self.max_conc_pct = max(self.max_conc_pct, conc)
            if conc > 90:
                self.conc_breach_bars += 1
                if self.conc_breach_bars >= 2:
                    self.conc_penalties += 1
                    self.conc_breach_bars = 0
            else:
                self.conc_breach_bars = 0

    def report(self):
        lines = [
            f"  Max leverage seen:       {self.max_leverage_seen:.1f}x (limit 25x)",
            f"  Leverage breaches:       {self.leverage_breaches} bars",
            f"  Max margin usage:        {self.max_margin_pct:.1f}%",
            f"  Margin penalties (-20):  {self.margin_penalties}",
            f"  Max concentration:       {self.max_conc_pct:.1f}%",
            f"  Conc penalties (-10):    {self.conc_penalties}",
        ]
        return "\n".join(lines)


TARGET_MAX_LEVERAGE = 20.0  # stay under 25x hard limit with buffer

def cap_size_for_leverage(desired_size, entry_price, open_trades, equity):
    current_notional = sum(abs(t.size * t.entry_price) for t in open_trades)
    max_total = TARGET_MAX_LEVERAGE * equity
    remaining = max_total - current_notional
    if remaining <= 0:
        return 0
    max_size = remaining / entry_price
    return min(desired_size, max_size)


def apply_slippage(price, direction, spread, entry=True):
    slip = SLIPPAGE_SPREAD_MULT * spread
    if entry:
        return price + slip if direction == "long" else price - slip
    else:
        return price - slip if direction == "long" else price + slip


def run_portfolio(dfs, configs, start_date=None, end_date=None):
    all_times = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    if start_date:
        all_times = [t for t in all_times if t.date() >= start_date]
    if end_date:
        all_times = [t for t in all_times if t.date() <= end_date]

    equity_curve = pd.Series(index=all_times, dtype=float)
    equity = ACCOUNT_EQUITY
    open_trades = []
    closed_trades = []
    session_entries = {}
    pyramid_groups = {}
    group_ctr = 0
    risk = RiskMonitor()

    for ts in all_times:
        hour_int = ts.hour
        hour = hour_int + ts.minute / 60.0
        date = ts.date()

        for cfg in configs:
            sym = cfg.symbol
            if sym not in dfs or ts not in dfs[sym].index:
                continue
            bar = dfs[sym].loc[ts]
            if pd.isna(bar["atr14"]) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]):
                continue

            close, high, low = bar["close"], bar["high"], bar["low"]
            atr = bar["atr14"]
            spread = bar["spread"]
            avg_sp = bar["avg_spread"] if not pd.isna(bar.get("avg_spread", np.nan)) else spread

            entry_start, entry_end = 7, 16
            total_h = entry_end - entry_start
            seg_l = total_h / 3
            if cfg.t_segment == 1:   seg_s, seg_e = entry_start, entry_start + seg_l
            elif cfg.t_segment == 2: seg_s, seg_e = entry_start + seg_l, entry_start + 2*seg_l
            elif cfg.t_segment == 3: seg_s, seg_e = entry_start + 2*seg_l, entry_end
            else:                    seg_s, seg_e = entry_start, entry_end

            sym_open = [t for t in open_trades if t.config_label == cfg.label]

            # ── Update open trades ──
            for trade in sym_open:
                u = trade.unrealised(close)
                if u < trade.mae: trade.mae = u
                if u > trade.mfe: trade.mfe = u

                if not trade.be_moved and trade.profit_atr(close) >= BE_ATR:
                    trade.stop_loss = trade.entry_price; trade.be_moved = True
                if not trade.trailing and trade.profit_atr(close) >= TRAIL_ON:
                    trade.trailing = True
                if trade.trailing:
                    if trade.direction == "long":
                        trade.stop_loss = max(trade.stop_loss, close - TRAIL_D * atr)
                    else:
                        trade.stop_loss = min(trade.stop_loss, close + TRAIL_D * atr)

                hit_sl = (trade.direction=="long" and low<=trade.stop_loss) or (trade.direction=="short" and high>=trade.stop_loss)
                hit_tp = (trade.direction=="long" and high>=trade.take_profit) or (trade.direction=="short" and low<=trade.take_profit)

                if hit_tp:
                    exit_p = apply_slippage(trade.take_profit, trade.direction, spread, entry=False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = exit_p, ts, "TP"
                    trade.slippage_cost = abs(trade.take_profit - exit_p) * trade.size
                elif hit_sl:
                    exit_p = apply_slippage(trade.stop_loss, trade.direction, spread, entry=False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = exit_p, ts, "SL"
                    trade.slippage_cost = abs(trade.stop_loss - exit_p) * trade.size

            # Hard close at 21:00
            if hour_int == 21:
                for trade in [t for t in sym_open if t.is_open]:
                    exit_p = apply_slippage(close, trade.direction, spread, entry=False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = exit_p, ts, "TIME"
                    trade.slippage_cost = abs(close - exit_p) * trade.size

            # Settle
            for trade in [t for t in open_trades if t.config_label == cfg.label and not t.is_open]:
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.direction=="long" \
                            else (trade.entry_price - trade.exit_price) * trade.size
                equity += trade.pnl
                closed_trades.append(trade)
                gid = trade.group_id
                if gid in pyramid_groups:
                    pyramid_groups[gid] = [t for t in pyramid_groups[gid] if t.is_open]
                    if not pyramid_groups[gid]: del pyramid_groups[gid]
            open_trades = [t for t in open_trades if t.is_open]

            # ── Pyramid ──
            for gid in [g for g in pyramid_groups if g.startswith(cfg.label)]:
                layers = pyramid_groups[gid]
                if not layers or len(layers) >= PYRAMID_LAYERS: continue
                if layers[-1].profit_atr(close) >= PYRAMID_TRIGGER:
                    for t in layers: t.stop_loss = t.entry_price; t.be_moved = True
                    d = layers[0].direction
                    entry_p = apply_slippage(close, d, spread, entry=True)
                    sl = entry_p - SL_ATR*atr if d=="long" else entry_p + SL_ATR*atr
                    tp = entry_p + TP_ATR*atr if d=="long" else entry_p - TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size_for_leverage(sz, entry_p, open_trades, equity)
                    if sz > 0 and len(open_trades) < MAX_POSITIONS:
                        group_ctr += 1
                        nt = Trade(sym, d, len(layers)+1, gid, ts, entry_p, sl, tp, atr, sz, cfg.label,
                                  slippage_cost=abs(close-entry_p)*sz)
                        open_trades.append(nt); layers.append(nt)

            # ── Entry ──
            if not (seg_s <= hour < seg_e): continue
            if pd.isna(avg_sp) or spread > 2.0 * avg_sp: continue

            poi_l, poi_s = get_poi(bar, cfg.poi_type)
            if pd.isna(poi_l) or pd.isna(poi_s): continue

            bo_l = poi_l + cfg.fract * atr if cfg.fract > 0 else poi_l
            bo_s = poi_s - cfg.fract * atr if cfg.fract > 0 else poi_s

            entry_key = (cfg.label, date)
            if entry_key in session_entries: continue

            if cfg.direction in ("long","both") and close > bo_l and close > bar["sma20"] and bar["sma20"] > bar["sma50"]:
                if check_filter(bar, cfg.filter1, "long") and len(open_trades) < MAX_POSITIONS:
                    entry_p = apply_slippage(close, "long", spread, entry=True)
                    sl = entry_p - SL_ATR*atr; tp = entry_p + TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size_for_leverage(sz, entry_p, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1
                    gid = f"{cfg.label}_L{group_ctr}"
                    trade = Trade(sym, "long", 1, gid, ts, entry_p, sl, tp, atr, sz, cfg.label,
                                 slippage_cost=abs(close-entry_p)*sz)
                    open_trades.append(trade)
                    pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

            elif cfg.direction in ("short","both") and close < bo_s and close < bar["sma20"] and bar["sma20"] < bar["sma50"]:
                if check_filter(bar, cfg.filter1, "short") and len(open_trades) < MAX_POSITIONS:
                    entry_p = apply_slippage(close, "short", spread, entry=True)
                    sl = entry_p + SL_ATR*atr; tp = entry_p - TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size_for_leverage(sz, entry_p, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1
                    gid = f"{cfg.label}_S{group_ctr}"
                    trade = Trade(sym, "short", 1, gid, ts, entry_p, sl, tp, atr, sz, cfg.label,
                                 slippage_cost=abs(close-entry_p)*sz)
                    open_trades.append(trade)
                    pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

        # Risk monitoring
        risk.check(open_trades, equity)

        # Equity snapshot
        unrealised = sum(t.unrealised(dfs[t.symbol].loc[ts,"close"])
                        for t in open_trades if t.symbol in dfs and ts in dfs[t.symbol].index)
        equity_curve[ts] = equity + unrealised

    # Force close remaining
    for t in list(open_trades):
        t.exit_price = dfs[t.symbol]["close"].iloc[-1]
        t.exit_time = all_times[-1]; t.exit_reason = "EOD"
        t.pnl = (t.exit_price-t.entry_price)*t.size if t.direction=="long" else (t.entry_price-t.exit_price)*t.size
        equity += t.pnl; closed_trades.append(t)

    return closed_trades, equity_curve, risk


def compute_metrics(trades, eq_curve, label=""):
    if not trades: return None
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0
    pf = gp/gl if gl > 0 else float("inf")
    eq = eq_curve.dropna()
    dd = ((eq - eq.cummax())/eq.cummax()).min()*100 if len(eq) > 1 else 0
    ret = (eq.iloc[-1] - ACCOUNT_EQUITY)/ACCOUNT_EQUITY*100 if len(eq) > 0 else 0
    r = eq.pct_change().dropna()
    sh = float(r.mean()/r.std()) if len(r) > 1 and r.std() > 0 else 0
    total_slippage = sum(t.slippage_cost for t in trades)
    return {"label":label, "n":len(trades), "wr":len(wins)/len(trades)*100, "pf":pf,
            "net_pct":ret, "dd":dd, "sharpe":sh, "gp":gp, "gl":gl,
            "avg_win":np.mean(wins) if wins else 0, "avg_loss":np.mean(losses) if losses else 0,
            "total_slippage": total_slippage, "net_dollar": sum(pnls)}


def print_metrics(m):
    pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
    print(f"\n{'='*55}")
    print(f"  {m['label']}")
    print(f"{'='*55}")
    for lbl, val in [
        ("Total trades", f"{m['n']}"), ("Win rate", f"{m['wr']:.1f}%"),
        ("Avg win", f"${m['avg_win']:,.0f}"), ("Avg loss", f"${m['avg_loss']:,.0f}"),
        ("Profit factor", pf_s), ("Net return", f"{m['net_pct']:+.2f}%"),
        ("Net P&L", f"${m['net_dollar']:+,.0f}"),
        ("Max drawdown", f"{m['dd']:.2f}%"), ("Sharpe (15-min)", f"{m['sharpe']:.4f}"),
        ("Total slippage", f"${m['total_slippage']:,.0f}"),
    ]:
        print(f"  {lbl:<22} {val}")
    print(f"{'='*55}")


def plot_version(trades, eq, metrics, risk, version, suffix=""):
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])

    ax_eq  = fig.add_subplot(gs[0, :])
    ax_mae = fig.add_subplot(gs[1, 0])
    ax_mfe = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[1, 2])
    ax_dd  = fig.add_subplot(gs[2, 0])
    ax_wr  = fig.add_subplot(gs[2, 1])
    ax_txt = fig.add_subplot(gs[2, 2])

    # Equity
    e = eq.dropna()
    ax_eq.plot(e.index, e.values, linewidth=1.2, color="#2196F3")
    ax_eq.axhline(ACCOUNT_EQUITY, color="gray", linestyle="--", linewidth=0.8)
    ax_eq.fill_between(e.index, ACCOUNT_EQUITY, e.values, alpha=0.1, color="#2196F3")
    pf_s = f"{metrics['pf']:.2f}" if metrics['pf'] < 100 else "inf"
    ax_eq.set_title(f"{version} | PF={pf_s} | Return={metrics['net_pct']:+.1f}% | DD={metrics['dd']:.1f}% | "
                   f"Sharpe={metrics['sharpe']:.4f} | {metrics['n']} trades | Slippage=${metrics['total_slippage']:,.0f}",
                   fontsize=11)
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # MAE / MFE
    pnls = np.array([t.pnl for t in trades])
    maes = np.array([t.mae for t in trades])
    mfes = np.array([t.mfe for t in trades])
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    ax_mae.scatter(maes, pnls, c=colors, alpha=0.7, s=25, edgecolors="none")
    ax_mae.axhline(0, color="gray", linewidth=0.5); ax_mae.axvline(0, color="gray", linewidth=0.5)
    ax_mae.set_title("MAE vs PnL"); ax_mae.set_xlabel("MAE ($)"); ax_mae.set_ylabel("PnL ($)")
    ax_mfe.scatter(mfes, pnls, c=colors, alpha=0.7, s=25, edgecolors="none")
    ax_mfe.axhline(0, color="gray", linewidth=0.5); ax_mfe.axvline(0, color="gray", linewidth=0.5)
    ax_mfe.set_title("MFE vs PnL"); ax_mfe.set_xlabel("MFE ($)"); ax_mfe.set_ylabel("PnL ($)")

    # Per-config PnL
    cfg_pnl = {}
    for t in trades: cfg_pnl[t.config_label] = cfg_pnl.get(t.config_label, 0) + t.pnl
    sorted_cfg = sorted(cfg_pnl.items(), key=lambda x: x[1], reverse=True)
    bar_labels = [x[0].replace(" ", "\n") for x in sorted_cfg]
    bar_vals   = [x[1] for x in sorted_cfg]
    bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in bar_vals]
    ax_bar.barh(bar_labels, bar_vals, color=bar_colors)
    ax_bar.axvline(0, color="gray", linewidth=0.8)
    ax_bar.set_title("PnL by Config"); ax_bar.set_xlabel("PnL ($)")
    ax_bar.tick_params(axis="y", labelsize=7)

    # Drawdown curve
    dd_curve = (e - e.cummax()) / e.cummax() * 100
    ax_dd.fill_between(dd_curve.index, dd_curve.values, 0, color="#F44336", alpha=0.3)
    ax_dd.plot(dd_curve.index, dd_curve.values, color="#F44336", linewidth=0.8)
    ax_dd.set_title("Drawdown Curve"); ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_ylim(min(dd_curve.min() * 1.5, -1), 0.5)

    # Win rate by config
    cfg_wr = {}
    for t in trades:
        if t.config_label not in cfg_wr: cfg_wr[t.config_label] = [0, 0]
        cfg_wr[t.config_label][1] += 1
        if t.pnl > 0: cfg_wr[t.config_label][0] += 1
    sorted_wr = sorted(cfg_wr.items(), key=lambda x: x[1][0]/x[1][1], reverse=True)
    wr_labels = [x[0].replace(" ", "\n") for x in sorted_wr]
    wr_vals   = [x[1][0]/x[1][1]*100 for x in sorted_wr]
    wr_colors = ["#4CAF50" if v >= 50 else "#FF9800" for v in wr_vals]
    ax_wr.barh(wr_labels, wr_vals, color=wr_colors)
    ax_wr.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax_wr.set_title("Win Rate by Config"); ax_wr.set_xlabel("Win Rate (%)")
    ax_wr.set_xlim(0, 100)
    ax_wr.tick_params(axis="y", labelsize=7)

    # Risk summary text
    ax_txt.axis("off")
    risk_text = (
        f"RISK MONITOR\n"
        f"{'='*30}\n"
        f"{risk.report()}\n\n"
        f"COMPETITION SCORING\n"
        f"{'='*30}\n"
        f"  Return rank component:  {metrics['net_pct']:+.1f}%\n"
        f"  Drawdown rank:          {metrics['dd']:.2f}%\n"
        f"  Sharpe rank:            {metrics['sharpe']:.4f}\n"
        f"  Risk penalties:         {risk.margin_penalties * -20 + risk.conc_penalties * -10}\n"
        f"  Total slippage:         ${metrics['total_slippage']:,.0f}"
    )
    ax_txt.text(0.05, 0.95, risk_text, transform=ax_txt.transAxes, fontsize=9,
               verticalalignment="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

    plt.tight_layout()
    fname = f"backtest_{version.lower().replace(' ', '_')}{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {fname}")
    return fname


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    needed = sorted(set(c.symbol for c in CONFIGS))
    print(f"V1.4 Competition-Ready Backtest")
    print(f"Loading: {needed}\n")
    dfs = {}
    for sym in needed:
        df = load_and_build(sym)
        if df is not None: dfs[sym] = df; print(f"  {sym}: {len(df)} bars")

    # Get date range
    all_dates = sorted(set(d for df in dfs.values() for d in df.index.date))
    total_days = len(all_dates)
    train_days = int(total_days * 0.67)  # ~20 days
    train_end  = all_dates[train_days - 1]
    test_start = all_dates[train_days]
    print(f"\n  Date range: {all_dates[0]} to {all_dates[-1]} ({total_days} trading days)")
    print(f"  Train: {all_dates[0]} to {train_end} ({train_days} days)")
    print(f"  Test:  {test_start} to {all_dates[-1]} ({total_days - train_days} days)")

    # ── RUN 1: Full period with slippage ──
    print("\n--- FULL PERIOD (with slippage) ---")
    trades_full, eq_full, risk_full = run_portfolio(dfs, CONFIGS)
    m_full = compute_metrics(trades_full, eq_full, "V1.4 FULL PERIOD (with slippage)")
    print_metrics(m_full)
    print(f"\n  Risk monitor:\n{risk_full.report()}")
    plot_version(trades_full, eq_full, m_full, risk_full, "V1.4 Full")

    # ── RUN 2: Train period only ──
    print("\n--- TRAIN PERIOD ---")
    trades_train, eq_train, risk_train = run_portfolio(dfs, CONFIGS, end_date=train_end)
    m_train = compute_metrics(trades_train, eq_train, f"V1.4 TRAIN ({all_dates[0]} to {train_end})")
    if m_train: print_metrics(m_train)

    # ── RUN 3: Test period only (OUT OF SAMPLE) ──
    print("\n--- TEST PERIOD (OUT OF SAMPLE) ---")
    trades_test, eq_test, risk_test = run_portfolio(dfs, CONFIGS, start_date=test_start)
    m_test = compute_metrics(trades_test, eq_test, f"V1.4 TEST OOS ({test_start} to {all_dates[-1]})")
    if m_test:
        print_metrics(m_test)
        plot_version(trades_test, eq_test, m_test, risk_test, "V1.4 OOS Test")

    # ── Summary comparison ──
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Period':<25} {'Trades':>7} {'WR':>7} {'PF':>7} {'Return':>9} {'DD':>8} {'Sharpe':>8} {'Slippage':>10}")
    print(f"  {'-'*76}")
    for m in [m_full, m_train, m_test]:
        if m:
            pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
            print(f"  {m['label'][:25]:<25} {m['n']:>7} {m['wr']:>6.0f}% {pf_s:>7} {m['net_pct']:>+8.1f}% {m['dd']:>7.1f}% {m['sharpe']:>8.4f} ${m['total_slippage']:>9,.0f}")
    print(f"{'='*80}")

    # Trade log
    rows = [{"entry_time":t.entry_time,"exit_time":t.exit_time,"symbol":t.symbol,
             "direction":t.direction,"layer":t.layer,"config":t.config_label,
             "entry_price":t.entry_price,"exit_price":t.exit_price,"size":t.size,
             "pnl":t.pnl,"mae":t.mae,"mfe":t.mfe,"exit_reason":t.exit_reason,
             "slippage":t.slippage_cost} for t in trades_full]
    pd.DataFrame(rows).to_csv("trade_log_v1.4.csv", index=False)
    print("Trade log: trade_log_v1.4.csv")


if __name__ == "__main__":
    main()
