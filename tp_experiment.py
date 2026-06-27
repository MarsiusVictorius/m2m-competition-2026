"""
TP Experiment: Compare TP=1.6 vs TP=2.0 across walk-forward splits.
Quick test to see if tighter TP is genuinely better.
"""

import os, glob, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR       = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
CANDLE_MINUTES = 15
ACCOUNT_EQUITY = 1_000_000.0
FIXED_RISK     = 10_000.0
MAX_POSITIONS  = 6
TARGET_MAX_LEVERAGE = 20.0
SLIPPAGE_SPREAD_MULT = 0.5
SL_ATR = 1.5; BE_ATR = 1.0; TRAIL_ON = 1.0; TRAIL_D = 1.0
PYRAMID_LAYERS = 3; PYRAMID_TRIGGER = 1.0

@dataclass
class BOSConfig:
    symbol: str; direction: str; poi_type: str = "asia"
    fract: float = 0.0; filter1: str = "none"; t_segment: int = 0; label: str = ""

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

def load_and_build(symbol):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{symbol}_*.parquet")))
    if not files: return None
    dfs = [pd.read_parquet(f, columns=["time","bid","ask"]) for f in files]
    ticks = pd.concat(dfs, ignore_index=True)
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    ticks = ticks.sort_values("time")
    ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2
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
    up_move = ohlc["high"] - ohlc["high"].shift(1)
    down_move = ohlc["low"].shift(1) - ohlc["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14_s = tr.ewm(span=14, adjust=False).mean()
    ohlc["dmi_plus"] = pd.Series(plus_dm, index=ohlc.index).ewm(span=14, adjust=False).mean() / atr14_s * 100
    ohlc["dmi_minus"] = pd.Series(minus_dm, index=ohlc.index).ewm(span=14, adjust=False).mean() / atr14_s * 100
    dx = (ohlc["dmi_plus"] - ohlc["dmi_minus"]).abs() / (ohlc["dmi_plus"] + ohlc["dmi_minus"]) * 100
    ohlc["adx"] = dx.ewm(span=14, adjust=False).mean()
    ohlc["ema20"] = ohlc["close"].ewm(span=20, adjust=False).mean()
    ohlc["_date"] = ohlc.index.date
    ohlc["_hour"] = ohlc.index.hour
    daily = ohlc.groupby("_date").agg(day_open=("open","first"), day_close=("close","last"))
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
    ohlc["lowest_l10"] = ohlc["low"].rolling(10).min()
    ohlc.drop(columns=["_date","_hour"], inplace=True)
    return ohlc

def get_poi(bar, poi_type):
    if poi_type == "asia": return bar.get("asia_high", np.nan), bar.get("asia_low", np.nan)
    elif poi_type == "prev_close":
        pc = bar.get("prev_close", np.nan); return pc, pc
    elif poi_type == "cum_ma":
        cm = bar.get("cum_ma", np.nan); return cm, cm
    elif poi_type == "ema20":
        e = bar.get("ema20", np.nan); return e, e
    return np.nan, np.nan

def check_filter(bar, filt, direction):
    if filt == "none": return True
    elif filt == "vol_expanding": return bar["atr5"] > bar["atr40"]
    elif filt == "vol_contracting": return bar["atr5"] <= bar["atr40"]
    elif filt == "dmi_trend":
        return bar["dmi_plus"] > bar["dmi_minus"] if direction=="long" else bar["dmi_minus"] > bar["dmi_plus"]
    elif filt == "adx_trending": return bar["adx"] > 25
    elif filt == "adx_ranging": return bar["adx"] < 25
    elif filt == "pullback_atr":
        if direction == "long": return bar["atr14"] > bar.get("prev_close", bar["close"]) - bar["lowest_l10"]
        return bar["atr14"] > bar["highest_h10"] - bar.get("prev_close", bar["close"])
    elif filt == "close_vs_open":
        return bar["close"] > bar.get("day_open", bar["close"]) if direction=="long" else bar["close"] < bar.get("day_open", bar["close"])
    return True

@dataclass
class Trade:
    symbol: str; direction: str; layer: int; group_id: str
    entry_time: object; entry_price: float; stop_loss: float; take_profit: float
    atr: float; size: float; config_label: str
    exit_time: object = None; exit_price: float = None
    pnl: float = 0.0; exit_reason: str = ""
    be_moved: bool = False; trailing: bool = False
    @property
    def is_open(self): return self.exit_time is None
    def unrealised(self, price):
        return (price - self.entry_price) * self.size if self.direction=="long" else (self.entry_price - price) * self.size
    def profit_atr(self, price):
        raw = (price - self.entry_price) if self.direction=="long" else (self.entry_price - price)
        return raw / self.atr if self.atr > 0 else 0

def cap_size(desired, entry_price, open_trades, equity):
    current = sum(abs(t.size * t.entry_price) for t in open_trades)
    remaining = TARGET_MAX_LEVERAGE * equity - current
    if remaining <= 0: return 0
    return min(desired, remaining / entry_price)

def apply_slip(price, direction, spread, entry=True):
    slip = SLIPPAGE_SPREAD_MULT * spread
    if entry: return price + slip if direction == "long" else price - slip
    else: return price - slip if direction == "long" else price + slip

def run_period(dfs, tp_atr, start_date=None, end_date=None):
    all_times = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    if start_date: all_times = [t for t in all_times if t.date() >= start_date]
    if end_date: all_times = [t for t in all_times if t.date() <= end_date]
    if not all_times: return {"n":0,"wr":0,"pf":0,"net_pct":0,"net_dollar":0}

    equity = ACCOUNT_EQUITY; open_trades = []; closed_trades = []
    session_entries = {}; pyramid_groups = {}; group_ctr = 0
    eq_vals = []

    for ts in all_times:
        hour_int = ts.hour; hour = hour_int + ts.minute / 60.0; date = ts.date()
        for cfg in CONFIGS:
            sym = cfg.symbol
            if sym not in dfs or ts not in dfs[sym].index: continue
            bar = dfs[sym].loc[ts]
            if pd.isna(bar["atr14"]) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]): continue
            close, high, low = bar["close"], bar["high"], bar["low"]
            atr = bar["atr14"]; spread = bar["spread"]
            avg_sp = bar["avg_spread"] if not pd.isna(bar.get("avg_spread", np.nan)) else spread
            entry_start, entry_end = 7, 16; total_h = entry_end - entry_start; seg_l = total_h / 3
            if cfg.t_segment == 1: seg_s, seg_e = entry_start, entry_start + seg_l
            elif cfg.t_segment == 2: seg_s, seg_e = entry_start + seg_l, entry_start + 2*seg_l
            elif cfg.t_segment == 3: seg_s, seg_e = entry_start + 2*seg_l, entry_end
            else: seg_s, seg_e = entry_start, entry_end
            sym_open = [t for t in open_trades if t.config_label == cfg.label]

            for trade in sym_open:
                if not trade.be_moved and trade.profit_atr(close) >= BE_ATR:
                    trade.stop_loss = trade.entry_price; trade.be_moved = True
                if not trade.trailing and trade.profit_atr(close) >= TRAIL_ON: trade.trailing = True
                if trade.trailing:
                    if trade.direction == "long": trade.stop_loss = max(trade.stop_loss, close - TRAIL_D * atr)
                    else: trade.stop_loss = min(trade.stop_loss, close + TRAIL_D * atr)
                hit_sl = (trade.direction=="long" and low<=trade.stop_loss) or (trade.direction=="short" and high>=trade.stop_loss)
                hit_tp = (trade.direction=="long" and high>=trade.take_profit) or (trade.direction=="short" and low<=trade.take_profit)
                if hit_tp:
                    ep = apply_slip(trade.take_profit, trade.direction, spread, False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = ep, ts, "TP"
                elif hit_sl:
                    ep = apply_slip(trade.stop_loss, trade.direction, spread, False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = ep, ts, "SL"
            if hour_int == 21:
                for trade in [t for t in sym_open if t.is_open]:
                    ep = apply_slip(close, trade.direction, spread, False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = ep, ts, "TIME"
            for trade in [t for t in open_trades if t.config_label == cfg.label and not t.is_open]:
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.direction=="long" \
                            else (trade.entry_price - trade.exit_price) * trade.size
                equity += trade.pnl; closed_trades.append(trade)
                gid = trade.group_id
                if gid in pyramid_groups:
                    pyramid_groups[gid] = [t for t in pyramid_groups[gid] if t.is_open]
                    if not pyramid_groups[gid]: del pyramid_groups[gid]
            open_trades = [t for t in open_trades if t.is_open]

            # Pyramid
            for gid in [g for g in pyramid_groups if g.startswith(cfg.label)]:
                layers = pyramid_groups[gid]
                if not layers or len(layers) >= PYRAMID_LAYERS: continue
                if layers[-1].profit_atr(close) >= PYRAMID_TRIGGER:
                    for t in layers: t.stop_loss = t.entry_price; t.be_moved = True
                    d = layers[0].direction
                    ep = apply_slip(close, d, spread, True)
                    sl = ep - SL_ATR*atr if d=="long" else ep + SL_ATR*atr
                    tp = ep + tp_atr*atr if d=="long" else ep - tp_atr*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz > 0 and len(open_trades) < MAX_POSITIONS:
                        group_ctr += 1
                        nt = Trade(sym, d, len(layers)+1, gid, ts, ep, sl, tp, atr, sz, cfg.label)
                        open_trades.append(nt); layers.append(nt)

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
                    ep = apply_slip(close, "long", spread, True)
                    sl = ep - SL_ATR*atr; tp = ep + tp_atr*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1; gid = f"{cfg.label}_L{group_ctr}"
                    trade = Trade(sym, "long", 1, gid, ts, ep, sl, tp, atr, sz, cfg.label)
                    open_trades.append(trade); pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True
            elif cfg.direction in ("short","both") and close < bo_s and close < bar["sma20"] and bar["sma20"] < bar["sma50"]:
                if check_filter(bar, cfg.filter1, "short") and len(open_trades) < MAX_POSITIONS:
                    ep = apply_slip(close, "short", spread, True)
                    sl = ep + SL_ATR*atr; tp = ep - tp_atr*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1; gid = f"{cfg.label}_S{group_ctr}"
                    trade = Trade(sym, "short", 1, gid, ts, ep, sl, tp, atr, sz, cfg.label)
                    open_trades.append(trade); pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

        unrealised = sum(t.unrealised(dfs[t.symbol].loc[ts,"close"])
                        for t in open_trades if t.symbol in dfs and ts in dfs[t.symbol].index)
        eq_vals.append(equity + unrealised)

    for t in list(open_trades):
        t.exit_price = dfs[t.symbol]["close"].iloc[-1]
        t.exit_time = all_times[-1]; t.exit_reason = "EOD"
        t.pnl = (t.exit_price-t.entry_price)*t.size if t.direction=="long" else (t.entry_price-t.exit_price)*t.size
        equity += t.pnl; closed_trades.append(t)

    pnls = [t.pnl for t in closed_trades]
    if not pnls: return {"n":0,"wr":0,"pf":0,"net_pct":0,"net_dollar":0,"eq":eq_vals}
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0
    pf = gp/gl if gl > 0 else float("inf")
    tp_hits = len([t for t in closed_trades if t.exit_reason == "TP"])
    sl_hits = len([t for t in closed_trades if t.exit_reason == "SL"])
    return {"n": len(pnls), "wr": len(wins)/len(pnls)*100, "pf": pf,
            "net_pct": sum(pnls)/ACCOUNT_EQUITY*100, "net_dollar": sum(pnls),
            "tp_hits": tp_hits, "sl_hits": sl_hits, "eq": eq_vals}

def main():
    needed = sorted(set(c.symbol for c in CONFIGS))
    print("Loading data...")
    dfs = {}
    for sym in needed:
        df = load_and_build(sym)
        if df is not None: dfs[sym] = df; print(f"  {sym}: {len(df)} bars")

    all_dates = sorted(set(d for df in dfs.values() for d in df.index.date))
    total_days = len(all_dates)

    tp_values = [1.4, 1.6, 1.8, 2.0, 2.2, 2.5]

    # Test each TP across full + OOS
    splits = [(60,40), (67,33), (75,25)]

    print(f"\n{'='*100}")
    print(f"  TP EXPERIMENT: Full period + 3 walk-forward splits")
    print(f"{'='*100}")

    all_results = {}
    for tp in tp_values:
        all_results[tp] = {}
        print(f"\n  TP = {tp:.1f} ATR:")

        # Full period
        m = run_period(dfs, tp)
        pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
        print(f"    FULL:  N={m['n']:>3} WR={m['wr']:>5.1f}% PF={pf_s:>6} Net={m['net_pct']:>+6.1f}% TP_hits={m.get('tp_hits',0)} SL_hits={m.get('sl_hits',0)}")
        all_results[tp]["full"] = m

        # Walk-forward splits (test portion only)
        for train_pct, test_pct in splits:
            cut = int(total_days * train_pct / 100)
            test_start = all_dates[cut]
            m_test = run_period(dfs, tp, start_date=test_start)
            pf_s = f"{m_test['pf']:.2f}" if m_test['pf'] < 100 else "inf"
            print(f"    OOS {train_pct}/{test_pct}: N={m_test['n']:>3} WR={m_test['wr']:>5.1f}% PF={pf_s:>6} Net={m_test['net_pct']:>+6.1f}%")
            all_results[tp][f"oos_{train_pct}"] = m_test

    # Summary comparison
    print(f"\n{'='*100}")
    print(f"  COMPARISON MATRIX")
    print(f"{'='*100}")
    print(f"  {'TP':>5} | {'Full Net%':>10} {'Full WR':>8} {'Full PF':>8} | {'OOS60 Net%':>11} {'OOS67 Net%':>11} {'OOS75 Net%':>11} | {'Avg OOS':>8}")
    print(f"  {'-'*95}")
    for tp in tp_values:
        full = all_results[tp]["full"]
        oos60 = all_results[tp]["oos_60"]
        oos67 = all_results[tp]["oos_67"]
        oos75 = all_results[tp]["oos_75"]
        avg_oos = np.mean([oos60["net_pct"], oos67["net_pct"], oos75["net_pct"]])
        pf_s = f"{full['pf']:.2f}" if full['pf'] < 100 else "inf"
        marker = " <-- CURRENT" if tp == 2.0 else " <-- BEST" if avg_oos == max(np.mean([all_results[t][f"oos_{s}"]["net_pct"] for s in [60,67,75]]) for t in tp_values) else ""
        print(f"  {tp:>5.1f} | {full['net_pct']:>+9.1f}% {full['wr']:>7.1f}% {pf_s:>8} | {oos60['net_pct']:>+10.1f}% {oos67['net_pct']:>+10.1f}% {oos75['net_pct']:>+10.1f}% | {avg_oos:>+7.1f}%{marker}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Net return by TP
    ax = axes[0]
    full_nets = [all_results[tp]["full"]["net_pct"] for tp in tp_values]
    avg_oos_nets = [np.mean([all_results[tp][f"oos_{s}"]["net_pct"] for s in [60,67,75]]) for tp in tp_values]
    ax.plot(tp_values, full_nets, "o-", color="#2196F3", linewidth=2, markersize=8, label="Full period")
    ax.plot(tp_values, avg_oos_nets, "s--", color="#FF9800", linewidth=2, markersize=8, label="Avg OOS")
    ax.axvline(2.0, color="red", linestyle=":", alpha=0.5, label="Current (2.0)")
    ax.set_xlabel("TP (ATR multiples)"); ax.set_ylabel("Net Return %")
    ax.set_title("Return vs TP Target"); ax.legend(); ax.grid(alpha=0.3)

    # Win rate by TP
    ax = axes[1]
    wrs = [all_results[tp]["full"]["wr"] for tp in tp_values]
    pfs = [min(all_results[tp]["full"]["pf"], 10) for tp in tp_values]
    ax.plot(tp_values, wrs, "o-", color="#4CAF50", linewidth=2, markersize=8, label="Win Rate %")
    ax.set_xlabel("TP (ATR multiples)"); ax.set_ylabel("Win Rate %", color="#4CAF50")
    ax2 = ax.twinx()
    ax2.plot(tp_values, pfs, "s--", color="#9C27B0", linewidth=2, markersize=8, label="PF (cap 10)")
    ax2.set_ylabel("Profit Factor", color="#9C27B0")
    ax.axvline(2.0, color="red", linestyle=":", alpha=0.5)
    ax.set_title("Win Rate & PF vs TP Target")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    # OOS consistency
    ax = axes[2]
    x = np.arange(len(tp_values))
    w = 0.25
    oos60_nets = [all_results[tp]["oos_60"]["net_pct"] for tp in tp_values]
    oos67_nets = [all_results[tp]["oos_67"]["net_pct"] for tp in tp_values]
    oos75_nets = [all_results[tp]["oos_75"]["net_pct"] for tp in tp_values]
    ax.bar(x - w, oos60_nets, w, label="OOS 60/40", color="#2196F3", alpha=0.8)
    ax.bar(x, oos67_nets, w, label="OOS 67/33", color="#FF9800", alpha=0.8)
    ax.bar(x + w, oos75_nets, w, label="OOS 75/25", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"TP={tp}" for tp in tp_values])
    ax.set_ylabel("OOS Net Return %"); ax.set_title("OOS Return by Split")
    ax.legend(); ax.axhline(0, color="gray", linewidth=0.5)

    plt.suptitle("TP Target Experiment — V1.4 with Slippage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("tp_experiment_v1.4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: tp_experiment_v1.4.png")

if __name__ == "__main__":
    main()
