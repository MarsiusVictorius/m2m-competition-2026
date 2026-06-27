"""
V1.3: BOS-enhanced portfolio backtest.
Uses best robust configs from the BOS scan, runs unified portfolio with pyramiding.
Saves versioned charts.
"""

import os, glob, warnings
from dataclasses import dataclass
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
RISK_PER_TRADE = 0.01
MAX_POSITIONS  = 6           # increased — we have 11 symbol/direction combos now
MAX_LEVERAGE   = 25.0
FIXED_RISK     = 10_000.0    # keep fixed sizing for now

SL_ATR  = 1.5
TP_ATR  = 2.0
BE_ATR  = 1.0
TRAIL_ON = 1.0
TRAIL_D  = 1.0
PYRAMID_LAYERS  = 3
PYRAMID_TRIGGER = 1.0  # ATR profit before adding


@dataclass
class BOSConfig:
    symbol:    str
    direction: str
    poi_type:  str   = "asia"
    fract:     float = 0.0
    filter1:   str   = "none"
    t_segment: int   = 0
    label:     str   = ""


# Best robust configs from BOS scan
CONFIGS = [
    BOSConfig("AUDJPY",  "long",  "asia",       0.0, "none",             0, "AUDJPY long asia"),
    BOSConfig("AUDNZD",  "long",  "asia",       0.0, "vol_contracting",  3, "AUDNZD long asia+volC+T3"),
    BOSConfig("AUDUSD",  "short", "asia",       0.0, "vol_contracting",  2, "AUDUSD short asia+volC+T2"),
    BOSConfig("AUDUSD",  "long",  "cum_ma",     0.8, "pullback_atr",     1, "AUDUSD long cumMA+pull+T1"),
    BOSConfig("EURGBP",  "short", "asia",       0.3, "none",             1, "EURGBP short asia+0.3+T1"),
    BOSConfig("EURJPY",  "short", "prev_close", 1.0, "pullback_atr",     0, "EURJPY short prevC+pull"),
    BOSConfig("GBPUSD",  "long",  "asia",       0.0, "none",             1, "GBPUSD long asia+T1"),
    BOSConfig("NZDUSD",  "long",  "asia",       0.0, "none",             1, "NZDUSD long asia+T1"),
    BOSConfig("USDCNH",  "long",  "asia",       0.0, "vol_contracting",  0, "USDCNH long asia+volC"),
    BOSConfig("USDJPY",  "short", "prev_close", 0.5, "vol_expanding",    1, "USDJPY short prevC+volE+T1"),
    BOSConfig("USDJPY",  "long",  "prev_close", 0.5, "vol_expanding",    2, "USDJPY long prevC+volE+T2"),
    BOSConfig("XAUKUSD", "short", "prev_close", 0.5, "none",             0, "XAUKUSD short prevC+0.5"),
]


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

    # DMI/ADX
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

    # Daily refs
    ohlc["_date"] = ohlc.index.date
    ohlc["_hour"] = ohlc.index.hour
    daily = ohlc.groupby("_date").agg(
        day_open=("open","first"), day_high=("high","max"), day_low=("low","min"), day_close=("close","last"))
    daily["prev_close"] = daily["day_close"].shift(1)
    ohlc = ohlc.join(daily[["day_open","prev_close"]], on="_date")

    asia = ohlc[ohlc["_hour"] < 7].groupby("_date").agg(asia_high=("high","max"), asia_low=("low","min"))
    ohlc = ohlc.join(asia, on="_date")

    # Cumulative MA
    cum_ma = []; count = 0; running = 0.0; prev_d = None
    for idx, row in ohlc.iterrows():
        d = row["_date"]
        if d != prev_d: count = 0; running = 0.0; prev_d = d
        count += 1; running += (row["high"]+row["low"])/2
        cum_ma.append(running/count)
    ohlc["cum_ma"] = cum_ma

    # Pullback ref
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

    @property
    def is_open(self): return self.exit_time is None
    def unrealised(self, price):
        return (price - self.entry_price) * self.size if self.direction=="long" else (self.entry_price - price) * self.size
    def profit_atr(self, price):
        raw = (price - self.entry_price) if self.direction=="long" else (self.entry_price - price)
        return raw / self.atr if self.atr > 0 else 0


def run_portfolio(dfs, configs):
    all_times = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    equity_curve = pd.Series(index=all_times, dtype=float)
    equity = ACCOUNT_EQUITY
    open_trades = []
    closed_trades = []
    session_entries = {}
    pyramid_groups = {}
    group_ctr = 0

    # Per-config state for confirmation etc
    confirm = {id(c): {"long":0, "short":0} for c in configs}

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
            spread, avg_sp = bar["spread"], bar["avg_spread"] if not pd.isna(bar.get("avg_spread", np.nan)) else bar["spread"]

            # T-segment boundaries
            entry_start, entry_end = 7, 16
            total_h = entry_end - entry_start
            seg_l = total_h / 3
            if cfg.t_segment == 1:   seg_s, seg_e = entry_start, entry_start + seg_l
            elif cfg.t_segment == 2: seg_s, seg_e = entry_start + seg_l, entry_start + 2*seg_l
            elif cfg.t_segment == 3: seg_s, seg_e = entry_start + 2*seg_l, entry_end
            else:                    seg_s, seg_e = entry_start, entry_end

            cfg_id = id(cfg)
            sym_open = [t for t in open_trades if t.config_label == cfg.label]

            # ── UPDATE open trades ──
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

                if hit_tp:   trade.exit_price, trade.exit_time, trade.exit_reason = trade.take_profit, ts, "TP"
                elif hit_sl: trade.exit_price, trade.exit_time, trade.exit_reason = trade.stop_loss, ts, "SL"

            # Hard close at 21:00
            if hour_int == 21:
                for trade in [t for t in sym_open if t.is_open]:
                    trade.exit_price, trade.exit_time, trade.exit_reason = close, ts, "TIME"

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
                    sl = close - SL_ATR*atr if d=="long" else close + SL_ATR*atr
                    tp = close + TP_ATR*atr if d=="long" else close - TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    if sz > 0 and len(open_trades) < MAX_POSITIONS:
                        group_ctr += 1
                        nt = Trade(sym, d, len(layers)+1, gid, ts, close, sl, tp, atr, sz, cfg.label)
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
                    sl = close - SL_ATR*atr; tp = close + TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    group_ctr += 1
                    gid = f"{cfg.label}_L{group_ctr}"
                    trade = Trade(sym, "long", 1, gid, ts, close, sl, tp, atr, sz, cfg.label)
                    open_trades.append(trade)
                    pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

            elif cfg.direction in ("short","both") and close < bo_s and close < bar["sma20"] and bar["sma20"] < bar["sma50"]:
                if check_filter(bar, cfg.filter1, "short") and len(open_trades) < MAX_POSITIONS:
                    sl = close + SL_ATR*atr; tp = close - TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    group_ctr += 1
                    gid = f"{cfg.label}_S{group_ctr}"
                    trade = Trade(sym, "short", 1, gid, ts, close, sl, tp, atr, sz, cfg.label)
                    open_trades.append(trade)
                    pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

        # Equity snapshot
        unrealised = sum(t.unrealised(dfs[t.symbol].loc[ts,"close"])
                        for t in open_trades if t.symbol in dfs and ts in dfs[t.symbol].index)
        equity_curve[ts] = equity + unrealised

    # Force close
    for t in list(open_trades):
        t.exit_price = dfs[t.symbol]["close"].iloc[-1]
        t.exit_time = all_times[-1]; t.exit_reason = "EOD"
        t.pnl = (t.exit_price-t.entry_price)*t.size if t.direction=="long" else (t.entry_price-t.exit_price)*t.size
        equity += t.pnl; closed_trades.append(t)

    return closed_trades, equity_curve


def compute_metrics(trades, eq_curve):
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0
    pf = gp/gl if gl > 0 else float("inf")
    eq = eq_curve.dropna()
    dd = ((eq - eq.cummax())/eq.cummax()).min()*100
    ret = (eq.iloc[-1] - ACCOUNT_EQUITY)/ACCOUNT_EQUITY*100
    r = eq.pct_change().dropna()
    sh = float(r.mean()/r.std()) if r.std() > 0 else 0
    return {"n":len(trades), "wr":len(wins)/len(trades)*100, "pf":pf,
            "net_pct":ret, "dd":dd, "sharpe":sh, "gp":gp, "gl":gl,
            "avg_win":np.mean(wins) if wins else 0, "avg_loss":np.mean(losses) if losses else 0}


def main():
    needed = sorted(set(c.symbol for c in CONFIGS))
    print(f"V1.3 BOS Portfolio\nLoading: {needed}")
    dfs = {}
    for sym in needed:
        df = load_and_build(sym)
        if df is not None: dfs[sym] = df; print(f"  {sym}: {len(df)} bars")

    print(f"\nConfigs ({len(CONFIGS)}):")
    for c in CONFIGS:
        print(f"  {c.label}")

    print("\nRunning...")
    trades, eq = run_portfolio(dfs, CONFIGS)
    m = compute_metrics(trades, eq)
    print(f"\n{'='*55}")
    print(f"  V1.3 BOS PORTFOLIO RESULTS")
    print(f"{'='*55}")
    for label, val in [
        ("Total trades", f"{m['n']}"),
        ("Win rate", f"{m['wr']:.1f}%"),
        ("Avg win", f"${m['avg_win']:,.0f}"),
        ("Avg loss", f"${m['avg_loss']:,.0f}"),
        ("Profit factor", f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"),
        ("Net return", f"{m['net_pct']:+.2f}%"),
        ("Max drawdown", f"{m['dd']:.2f}%"),
        ("Sharpe (15-min)", f"{m['sharpe']:.4f}"),
    ]:
        print(f"  {label:<22} {val}")
    print(f"{'='*55}")

    # Per-config breakdown
    labels = sorted(set(t.config_label for t in trades))
    print(f"\n{'='*90}")
    print(f"  PER-CONFIG BREAKDOWN")
    print(f"{'='*90}")
    print(f"  {'Config':<35} {'N':>4} {'WR':>6} {'PF':>6} {'Net$':>11}")
    print(f"  {'-'*86}")
    for lbl in labels:
        lt = [t for t in trades if t.config_label == lbl]
        pnls = [t.pnl for t in lt]
        w = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        pf = gp/gl if gl > 0 else float("inf")
        pf_s = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"  {lbl:<35} {len(lt):>4} {w/len(lt)*100:>5.0f}% {pf_s:>6} {sum(pnls):>+10,.0f}")
    print(f"{'='*90}")

    # Trade log
    rows = [{"entry_time":t.entry_time,"exit_time":t.exit_time,"symbol":t.symbol,
             "direction":t.direction,"layer":t.layer,"config":t.config_label,
             "entry_price":t.entry_price,"exit_price":t.exit_price,"size":t.size,
             "pnl":t.pnl,"mae":t.mae,"mfe":t.mfe,"exit_reason":t.exit_reason} for t in trades]
    pd.DataFrame(rows).to_csv("trade_log_v1.3.csv", index=False)
    print("Trade log: trade_log_v1.3.csv")

    # ── V1.3 Chart ──
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax_eq  = fig.add_subplot(gs[0, :])
    ax_mae = fig.add_subplot(gs[1, 0])
    ax_mfe = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[1, 2])

    # Equity
    e = eq.dropna()
    ax_eq.plot(e.index, e.values, linewidth=1.2, color="#2196F3")
    ax_eq.axhline(ACCOUNT_EQUITY, color="gray", linestyle="--", linewidth=0.8)
    ax_eq.set_title(f"V1.3 BOS Portfolio | PF={m['pf']:.2f} | Return={m['net_pct']:+.1f}% | DD={m['dd']:.1f}% | Sharpe={m['sharpe']:.4f} | {m['n']} trades")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # MAE/MFE
    pnls = np.array([t.pnl for t in trades])
    maes = np.array([t.mae for t in trades])
    mfes = np.array([t.mfe for t in trades])
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    ax_mae.scatter(maes, pnls, c=colors, alpha=0.7, s=30, edgecolors="none")
    ax_mae.axhline(0, color="gray", linewidth=0.5); ax_mae.axvline(0, color="gray", linewidth=0.5)
    ax_mae.set_title("MAE vs PnL"); ax_mae.set_xlabel("MAE ($)"); ax_mae.set_ylabel("PnL ($)")
    ax_mfe.scatter(mfes, pnls, c=colors, alpha=0.7, s=30, edgecolors="none")
    ax_mfe.axhline(0, color="gray", linewidth=0.5); ax_mfe.axvline(0, color="gray", linewidth=0.5)
    ax_mfe.set_title("MFE vs PnL"); ax_mfe.set_xlabel("MFE ($)"); ax_mfe.set_ylabel("PnL ($)")

    # Per-config PnL bar
    cfg_pnl = {}
    for t in trades:
        cfg_pnl[t.config_label] = cfg_pnl.get(t.config_label, 0) + t.pnl
    sorted_cfg = sorted(cfg_pnl.items(), key=lambda x: x[1], reverse=True)
    bar_labels = [x[0].replace(" ", "\n") for x in sorted_cfg]
    bar_vals   = [x[1] for x in sorted_cfg]
    bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in bar_vals]
    ax_bar.barh(bar_labels, bar_vals, color=bar_colors)
    ax_bar.axvline(0, color="gray", linewidth=0.8)
    ax_bar.set_title("PnL by Config")
    ax_bar.set_xlabel("PnL ($)")
    ax_bar.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig("backtest_v1.3_bos_portfolio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart: backtest_v1.3_bos_portfolio.png")


if __name__ == "__main__":
    main()
