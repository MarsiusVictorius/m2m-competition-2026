"""
V1.4 Parameter Sensitivity Test
Wiggles key parameters ±20% to check strategy robustness.
Tests: SL_ATR, TP_ATR, BE_ATR, TRAIL_ON, TRAIL_D
"""

import os, glob, warnings, itertools
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
FIXED_RISK     = 10_000.0
MAX_POSITIONS  = 6
TARGET_MAX_LEVERAGE = 20.0
SLIPPAGE_SPREAD_MULT = 0.5
PYRAMID_LAYERS  = 3
PYRAMID_TRIGGER = 1.0


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
    if poi_type == "asia":
        return bar.get("asia_high", np.nan), bar.get("asia_low", np.nan)
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
        if direction == "long":
            return bar["atr14"] > bar.get("prev_close", bar["close"]) - bar["lowest_l10"]
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


def cap_size(desired, entry_price, open_trades, equity):
    current = sum(abs(t.size * t.entry_price) for t in open_trades)
    remaining = TARGET_MAX_LEVERAGE * equity - current
    if remaining <= 0: return 0
    return min(desired, remaining / entry_price)


def apply_slip(price, direction, spread, entry=True):
    slip = SLIPPAGE_SPREAD_MULT * spread
    if entry: return price + slip if direction == "long" else price - slip
    else: return price - slip if direction == "long" else price + slip


def run_with_params(dfs, sl_atr, tp_atr, be_atr, trail_on, trail_d):
    all_times = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    equity = ACCOUNT_EQUITY
    open_trades = []; closed_trades = []; session_entries = {}
    pyramid_groups = {}; group_ctr = 0

    for ts in all_times:
        hour_int = ts.hour
        hour = hour_int + ts.minute / 60.0
        date = ts.date()

        for cfg in CONFIGS:
            sym = cfg.symbol
            if sym not in dfs or ts not in dfs[sym].index: continue
            bar = dfs[sym].loc[ts]
            if pd.isna(bar["atr14"]) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]): continue

            close, high, low = bar["close"], bar["high"], bar["low"]
            atr = bar["atr14"]; spread = bar["spread"]
            avg_sp = bar["avg_spread"] if not pd.isna(bar.get("avg_spread", np.nan)) else spread

            entry_start, entry_end = 7, 16
            total_h = entry_end - entry_start; seg_l = total_h / 3
            if cfg.t_segment == 1:   seg_s, seg_e = entry_start, entry_start + seg_l
            elif cfg.t_segment == 2: seg_s, seg_e = entry_start + seg_l, entry_start + 2*seg_l
            elif cfg.t_segment == 3: seg_s, seg_e = entry_start + 2*seg_l, entry_end
            else:                    seg_s, seg_e = entry_start, entry_end

            sym_open = [t for t in open_trades if t.config_label == cfg.label]

            for trade in sym_open:
                u = trade.unrealised(close)
                if u < trade.mae: trade.mae = u
                if u > trade.mfe: trade.mfe = u
                if not trade.be_moved and trade.profit_atr(close) >= be_atr:
                    trade.stop_loss = trade.entry_price; trade.be_moved = True
                if not trade.trailing and trade.profit_atr(close) >= trail_on:
                    trade.trailing = True
                if trade.trailing:
                    if trade.direction == "long":
                        trade.stop_loss = max(trade.stop_loss, close - trail_d * atr)
                    else:
                        trade.stop_loss = min(trade.stop_loss, close + trail_d * atr)
                hit_sl = (trade.direction=="long" and low<=trade.stop_loss) or (trade.direction=="short" and high>=trade.stop_loss)
                hit_tp = (trade.direction=="long" and high>=trade.take_profit) or (trade.direction=="short" and low<=trade.take_profit)
                if hit_tp:
                    ep = apply_slip(trade.take_profit, trade.direction, spread, False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = ep, ts, "TP"
                    trade.slippage_cost = abs(trade.take_profit - ep) * trade.size
                elif hit_sl:
                    ep = apply_slip(trade.stop_loss, trade.direction, spread, False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = ep, ts, "SL"
                    trade.slippage_cost = abs(trade.stop_loss - ep) * trade.size

            if hour_int == 21:
                for trade in [t for t in sym_open if t.is_open]:
                    ep = apply_slip(close, trade.direction, spread, False)
                    trade.exit_price, trade.exit_time, trade.exit_reason = ep, ts, "TIME"
                    trade.slippage_cost = abs(close - ep) * trade.size

            for trade in [t for t in open_trades if t.config_label == cfg.label and not t.is_open]:
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if trade.direction=="long" \
                            else (trade.entry_price - trade.exit_price) * trade.size
                equity += trade.pnl; closed_trades.append(trade)
                gid = trade.group_id
                if gid in pyramid_groups:
                    pyramid_groups[gid] = [t for t in pyramid_groups[gid] if t.is_open]
                    if not pyramid_groups[gid]: del pyramid_groups[gid]
            open_trades = [t for t in open_trades if t.is_open]

            for gid in [g for g in pyramid_groups if g.startswith(cfg.label)]:
                layers = pyramid_groups[gid]
                if not layers or len(layers) >= PYRAMID_LAYERS: continue
                if layers[-1].profit_atr(close) >= PYRAMID_TRIGGER:
                    for t in layers: t.stop_loss = t.entry_price; t.be_moved = True
                    d = layers[0].direction
                    ep = apply_slip(close, d, spread, True)
                    sl = ep - sl_atr*atr if d=="long" else ep + sl_atr*atr
                    tp = ep + tp_atr*atr if d=="long" else ep - tp_atr*atr
                    sz = FIXED_RISK / (sl_atr * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz > 0 and len(open_trades) < MAX_POSITIONS:
                        group_ctr += 1
                        nt = Trade(sym, d, len(layers)+1, gid, ts, ep, sl, tp, atr, sz, cfg.label,
                                  slippage_cost=abs(close-ep)*sz)
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
                    sl = ep - sl_atr*atr; tp = ep + tp_atr*atr
                    sz = FIXED_RISK / (sl_atr * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1; gid = f"{cfg.label}_L{group_ctr}"
                    trade = Trade(sym, "long", 1, gid, ts, ep, sl, tp, atr, sz, cfg.label,
                                 slippage_cost=abs(close-ep)*sz)
                    open_trades.append(trade); pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

            elif cfg.direction in ("short","both") and close < bo_s and close < bar["sma20"] and bar["sma20"] < bar["sma50"]:
                if check_filter(bar, cfg.filter1, "short") and len(open_trades) < MAX_POSITIONS:
                    ep = apply_slip(close, "short", spread, True)
                    sl = ep + sl_atr*atr; tp = ep - tp_atr*atr
                    sz = FIXED_RISK / (sl_atr * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1; gid = f"{cfg.label}_S{group_ctr}"
                    trade = Trade(sym, "short", 1, gid, ts, ep, sl, tp, atr, sz, cfg.label,
                                 slippage_cost=abs(close-ep)*sz)
                    open_trades.append(trade); pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

    for t in list(open_trades):
        t.exit_price = dfs[t.symbol]["close"].iloc[-1]
        t.exit_time = all_times[-1]; t.exit_reason = "EOD"
        t.pnl = (t.exit_price-t.entry_price)*t.size if t.direction=="long" else (t.entry_price-t.exit_price)*t.size
        equity += t.pnl; closed_trades.append(t)

    pnls = [t.pnl for t in closed_trades]
    if not pnls: return {"n":0, "wr":0, "pf":0, "net_pct":0, "dd":0}
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0
    pf = gp/gl if gl > 0 else float("inf")
    net_pct = sum(pnls) / ACCOUNT_EQUITY * 100
    return {"n": len(pnls), "wr": len(wins)/len(pnls)*100, "pf": pf, "net_pct": net_pct, "net_dollar": sum(pnls)}


def main():
    needed = sorted(set(c.symbol for c in CONFIGS))
    print("Loading data...")
    dfs = {}
    for sym in needed:
        df = load_and_build(sym)
        if df is not None: dfs[sym] = df; print(f"  {sym}: {len(df)} bars")

    # Base params
    BASE = {"sl_atr": 1.5, "tp_atr": 2.0, "be_atr": 1.0, "trail_on": 1.0, "trail_d": 1.0}
    WIGGLE = [0.8, 0.9, 1.0, 1.1, 1.2]  # -20%, -10%, base, +10%, +20%

    results = []

    # Test each parameter independently
    params_to_test = ["sl_atr", "tp_atr", "be_atr", "trail_on", "trail_d"]

    for param in params_to_test:
        print(f"\nTesting {param}...")
        for mult in WIGGLE:
            test_params = BASE.copy()
            test_params[param] = BASE[param] * mult
            val = test_params[param]
            label = f"{param}={val:.2f} ({mult:.0%})"
            print(f"  {label}...", end=" ", flush=True)
            m = run_with_params(dfs, test_params["sl_atr"], test_params["tp_atr"],
                               test_params["be_atr"], test_params["trail_on"], test_params["trail_d"])
            pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
            print(f"N={m['n']}, WR={m['wr']:.0f}%, PF={pf_s}, Net={m['net_pct']:+.1f}%")
            results.append({"param": param, "mult": mult, "value": val, **m})

    # Summary table
    print(f"\n{'='*90}")
    print(f"  PARAMETER SENSITIVITY SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Parameter':<12} {'Mult':>6} {'Value':>7} {'Trades':>7} {'WR':>6} {'PF':>7} {'Net%':>8} {'Net$':>12}")
    print(f"  {'-'*82}")
    for r in results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        marker = " <-- BASE" if r["mult"] == 1.0 else ""
        print(f"  {r['param']:<12} {r['mult']:>5.0%} {r['value']:>7.2f} {r['n']:>7} {r['wr']:>5.0f}% {pf_s:>7} {r['net_pct']:>+7.1f}% ${r.get('net_dollar',0):>+10,.0f}{marker}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, param in enumerate(params_to_test):
        ax = axes[i]
        param_results = [r for r in results if r["param"] == param]
        mults = [r["mult"] for r in param_results]
        vals = [r["value"] for r in param_results]
        nets = [r["net_pct"] for r in param_results]
        pfs = [min(r["pf"], 10) for r in param_results]
        wrs = [r["wr"] for r in param_results]

        color_net = "#2196F3"
        color_pf = "#4CAF50"
        color_wr = "#FF9800"

        ax.plot(vals, nets, "o-", color=color_net, linewidth=2, markersize=8, label="Net Return %")
        ax.set_xlabel(f"{param} value")
        ax.set_ylabel("Net Return %", color=color_net)
        ax.tick_params(axis="y", labelcolor=color_net)

        ax2 = ax.twinx()
        ax2.plot(vals, pfs, "s--", color=color_pf, linewidth=1.5, markersize=6, label="PF (capped 10)")
        ax2.set_ylabel("Profit Factor", color=color_pf)
        ax2.tick_params(axis="y", labelcolor=color_pf)

        base_val = BASE[param]
        ax.axvline(base_val, color="red", linestyle=":", alpha=0.7, label="Base value")
        ax.set_title(f"{param} sensitivity", fontsize=12, fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    # Overall robustness summary in last subplot
    ax = axes[5]
    ax.axis("off")

    all_nets = [r["net_pct"] for r in results]
    all_pfs = [r["pf"] for r in results]
    base_results = [r for r in results if r["mult"] == 1.0]
    base_net = np.mean([r["net_pct"] for r in base_results])

    min_net = min(all_nets)
    max_net = max(all_nets)
    std_net = np.std(all_nets)
    all_positive = all(n > 0 for n in all_nets)
    all_pf_above_1 = all(p > 1.0 for p in all_pfs)

    robustness = "ROBUST" if all_positive and all_pf_above_1 and std_net < base_net * 0.5 else \
                 "MODERATE" if all_positive and std_net < base_net else "FRAGILE"

    summary_text = (
        f"ROBUSTNESS VERDICT: {robustness}\n"
        f"{'='*35}\n\n"
        f"Base net return:    {base_net:+.1f}%\n"
        f"Min across tests:   {min_net:+.1f}%\n"
        f"Max across tests:   {max_net:+.1f}%\n"
        f"Std dev:            {std_net:.1f}%\n\n"
        f"All variants profitable: {'YES' if all_positive else 'NO'}\n"
        f"All PF > 1.0:           {'YES' if all_pf_above_1 else 'NO'}\n\n"
        f"Tested: 5 params x 5 levels\n"
        f"= 25 parameter combinations\n"
        f"Wiggle range: +/- 20%"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="#e8f5e9" if robustness=="ROBUST" else "#fff3e0" if robustness=="MODERATE" else "#ffebee", alpha=0.9))

    plt.suptitle("V1.4 Parameter Sensitivity Test (+/- 20%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sensitivity_v1.4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: sensitivity_v1.4.png")


if __name__ == "__main__":
    main()
