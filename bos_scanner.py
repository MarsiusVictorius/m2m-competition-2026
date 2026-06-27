"""
BOS-enhanced edge scanner.
Ports Nesnidal's modular breakout framework into our FX/metals backtester:
  - Multiple POI types (12 variants)
  - ATR-fraction distance for breakout levels
  - Volatility regime filters
  - DMI/ADX trend filters
  - Pullback / price action filters
  - T-segmentation (session thirds)
  - Direction proximity guard
"""

import os, glob, warnings, itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

DATA_DIR       = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
CANDLE_MINUTES = 15
FIXED_RISK     = 10_000.0
ACCOUNT_EQUITY = 1_000_000.0

# Only test symbols where we found at least marginal edge
SYMBOLS = ["AUDJPY", "AUDNZD", "AUDUSD", "EURJPY", "EURGBP",
           "GBPUSD", "NZDUSD", "USDCNH", "USDJPY", "XAUKUSD"]

SL_ATR   = 1.5
TP_ATR   = 2.0
BE_ATR   = 1.0
TRAIL_ON = 1.0
TRAIL_D  = 1.0


# ─────────────────────────────────────────
# DATA LOADING + FULL INDICATOR SET
# ─────────────────────────────────────────
def load_and_build(symbol):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{symbol}_*.parquet")))
    if not files:
        return None
    dfs = [pd.read_parquet(f, columns=["time", "bid", "ask"]) for f in files]
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

    # Core indicators
    ohlc["sma20"] = ohlc["close"].rolling(20).mean()
    ohlc["sma50"] = ohlc["close"].rolling(50).mean()
    prev = ohlc["close"].shift(1)
    tr = pd.concat([ohlc["high"]-ohlc["low"], (ohlc["high"]-prev).abs(), (ohlc["low"]-prev).abs()], axis=1).max(axis=1)
    ohlc["atr14"]  = tr.rolling(14).mean()
    ohlc["atr5"]   = tr.rolling(5).mean()
    ohlc["atr40"]  = tr.rolling(40).mean()
    ohlc["avg_spread"] = ohlc["spread"].rolling(20).mean()

    # DMI components (14-period)
    up_move   = ohlc["high"] - ohlc["high"].shift(1)
    down_move = ohlc["low"].shift(1) - ohlc["low"]
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14_s   = tr.ewm(span=14, adjust=False).mean()
    ohlc["dmi_plus"]  = pd.Series(plus_dm, index=ohlc.index).ewm(span=14, adjust=False).mean() / atr14_s * 100
    ohlc["dmi_minus"] = pd.Series(minus_dm, index=ohlc.index).ewm(span=14, adjust=False).mean() / atr14_s * 100
    dx = (ohlc["dmi_plus"] - ohlc["dmi_minus"]).abs() / (ohlc["dmi_plus"] + ohlc["dmi_minus"]) * 100
    ohlc["adx"] = dx.ewm(span=14, adjust=False).mean()

    # EMA for POI types
    ohlc["ema20"] = ohlc["close"].ewm(span=20, adjust=False).mean()
    ohlc["ema_low20"]  = ohlc["low"].ewm(span=20, adjust=False).mean()
    ohlc["ema_high20"] = ohlc["high"].ewm(span=20, adjust=False).mean()

    # Bollinger Bands (20, 1.5 std)
    ohlc["bb_upper"] = ohlc["close"].rolling(20).mean() + 1.5 * ohlc["close"].rolling(20).std()
    ohlc["bb_lower"] = ohlc["close"].rolling(20).mean() - 1.5 * ohlc["close"].rolling(20).std()

    # Daily reference levels
    ohlc["_date"] = ohlc.index.date
    ohlc["_hour"] = ohlc.index.hour

    # Previous day close, today open, today high/low, prev day high/low
    daily = ohlc.groupby("_date").agg(
        day_open=("open", "first"), day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")
    )
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["prev_high"]  = daily["day_high"].shift(1)
    daily["prev_low"]   = daily["day_low"].shift(1)
    ohlc = ohlc.join(daily[["day_open", "day_high", "day_low", "prev_close", "prev_high", "prev_low"]], on="_date")

    # Asia session range
    asia = ohlc[ohlc["_hour"] < 7].groupby("_date").agg(asia_high=("high","max"), asia_low=("low","min"))
    ohlc = ohlc.join(asia, on="_date")

    # Cumulative intraday MA
    ohlc["mid_hl"] = (ohlc["high"] + ohlc["low"]) / 2
    cum_ma = []
    count = 0
    running_sum = 0.0
    prev_date = None
    for idx, row in ohlc.iterrows():
        d = row["_date"]
        if d != prev_date:
            count = 0
            running_sum = 0.0
            prev_date = d
        count += 1
        running_sum += row["mid_hl"]
        cum_ma.append(running_sum / count)
    ohlc["cum_ma"] = cum_ma

    # Highest/Lowest N bars (for pullback filters)
    ohlc["highest_h10"] = ohlc["high"].rolling(10).max()
    ohlc["lowest_l10"]  = ohlc["low"].rolling(10).min()
    ohlc["highest_h20"] = ohlc["high"].rolling(20).max()
    ohlc["lowest_l20"]  = ohlc["low"].rolling(20).min()

    ohlc.drop(columns=["_date", "_hour", "mid_hl"], inplace=True)
    return ohlc


# ─────────────────────────────────────────
# POI TYPES (from BOS)
# ─────────────────────────────────────────
def get_poi(bar, poi_type):
    """Return (poi_long, poi_short) for the given POI type."""
    if poi_type == "asia":
        return bar.get("asia_high", np.nan), bar.get("asia_low", np.nan)
    elif poi_type == "prev_close":
        pc = bar.get("prev_close", np.nan)
        return pc, pc
    elif poi_type == "day_open":
        do = bar.get("day_open", np.nan)
        return do, do
    elif poi_type == "day_lohi":
        return bar.get("day_low", np.nan), bar.get("day_high", np.nan)
    elif poi_type == "prev_lohi":
        return bar.get("prev_low", np.nan), bar.get("prev_high", np.nan)
    elif poi_type == "minmax_pc_do":
        pc = bar.get("prev_close", np.nan)
        do = bar.get("day_open", np.nan)
        if pd.isna(pc) or pd.isna(do):
            return np.nan, np.nan
        return min(pc, do), max(pc, do)
    elif poi_type == "maxmin_pc_do":
        pc = bar.get("prev_close", np.nan)
        do = bar.get("day_open", np.nan)
        if pd.isna(pc) or pd.isna(do):
            return np.nan, np.nan
        return max(pc, do), min(pc, do)
    elif poi_type == "cum_ma":
        cm = bar.get("cum_ma", np.nan)
        return cm, cm
    elif poi_type == "ema20":
        e = bar.get("ema20", np.nan)
        return e, e
    elif poi_type == "ema_lohi":
        return bar.get("ema_low20", np.nan), bar.get("ema_high20", np.nan)
    elif poi_type == "bb":
        return bar.get("bb_lower", np.nan), bar.get("bb_upper", np.nan)
    return np.nan, np.nan


# ─────────────────────────────────────────
# FILTER FUNCTIONS
# ─────────────────────────────────────────
def check_filter(bar, filt, direction):
    """Check if filter passes for given direction. Returns True if pass."""
    if filt == "none":
        return True

    # Volatility regime filters
    elif filt == "vol_expanding":
        return bar["atr5"] > bar["atr40"]
    elif filt == "vol_contracting":
        return bar["atr5"] <= bar["atr40"]

    # DMI filters
    elif filt == "dmi_trend":
        if direction == "long":
            return bar["dmi_plus"] > bar["dmi_minus"]
        return bar["dmi_minus"] > bar["dmi_plus"]
    elif filt == "dmi_counter":
        if direction == "long":
            return bar["dmi_plus"] < bar["dmi_minus"]
        return bar["dmi_minus"] < bar["dmi_plus"]

    # ADX filters
    elif filt == "adx_trending":
        return bar["adx"] > 25
    elif filt == "adx_ranging":
        return bar["adx"] < 25

    # Pullback filters
    elif filt == "pullback_atr":
        if direction == "long":
            return bar["atr14"] > bar.get("prev_close", bar["close"]) - bar["lowest_l10"]
        return bar["atr14"] > bar["highest_h10"] - bar.get("prev_close", bar["close"])

    # Price action filters
    elif filt == "close_vs_open":
        if direction == "long":
            return bar["close"] > bar.get("day_open", bar["close"])
        return bar["close"] < bar.get("day_open", bar["close"])
    elif filt == "close_vs_open_counter":
        if direction == "long":
            return bar["close"] < bar.get("day_open", bar["close"])
        return bar["close"] > bar.get("day_open", bar["close"])

    return True


# ─────────────────────────────────────────
# STRATEGY CONFIG
# ─────────────────────────────────────────
@dataclass
class BOSConfig:
    direction:      str   = "both"
    poi_type:       str   = "asia"
    fract:          float = 0.0     # 0 = use POI directly (for asia/lohi types), >0 = add fract*ATR
    filter1:        str   = "none"
    t_segment:      int   = 0       # 0=full, 1=first third, 2=middle, 3=last third
    proximity_guard:bool  = True
    entry_start:    int   = 7
    entry_end:      int   = 16


# ─────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────
def run_bos(df, cfg: BOSConfig):
    trades = []
    in_trade = None
    session_used = {}

    # T-segment boundaries
    total_hours = cfg.entry_end - cfg.entry_start
    seg_len = total_hours / 3
    if cfg.t_segment == 1:
        seg_start, seg_end = cfg.entry_start, cfg.entry_start + seg_len
    elif cfg.t_segment == 2:
        seg_start, seg_end = cfg.entry_start + seg_len, cfg.entry_start + 2*seg_len
    elif cfg.t_segment == 3:
        seg_start, seg_end = cfg.entry_start + 2*seg_len, cfg.entry_end
    else:
        seg_start, seg_end = cfg.entry_start, cfg.entry_end

    for ts, bar in df.iterrows():
        hour = ts.hour + ts.minute / 60.0
        hour_int = ts.hour
        date = ts.date()
        close, high, low = bar["close"], bar["high"], bar["low"]
        atr = bar["atr14"]
        spread, avg_sp = bar["spread"], bar["avg_spread"]

        if pd.isna(atr) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]):
            continue

        # ── Manage open trade ──
        if in_trade is not None:
            t = in_trade
            profit_atr = (close - t["entry"]) / t["atr"] if t["dir"] == "long" else (t["entry"] - close) / t["atr"]
            if not t["be"] and profit_atr >= BE_ATR:
                t["sl"] = t["entry"]; t["be"] = True
            if not t["trailing"] and profit_atr >= TRAIL_ON:
                t["trailing"] = True
            if t["trailing"]:
                if t["dir"] == "long":  t["sl"] = max(t["sl"], close - TRAIL_D * atr)
                else:                   t["sl"] = min(t["sl"], close + TRAIL_D * atr)
            upnl = (close - t["entry"]) if t["dir"] == "long" else (t["entry"] - close)
            t["mae_price"] = min(t["mae_price"], upnl)
            t["mfe_price"] = max(t["mfe_price"], upnl)

            hit_sl = (t["dir"]=="long" and low<=t["sl"]) or (t["dir"]=="short" and high>=t["sl"])
            hit_tp = (t["dir"]=="long" and high>=t["tp"]) or (t["dir"]=="short" and low<=t["tp"])
            time_exit = (hour_int == 21)

            exit_price = reason = None
            if hit_tp:      exit_price, reason = t["tp"], "TP"
            elif hit_sl:    exit_price, reason = t["sl"], "SL"
            elif time_exit: exit_price, reason = close, "TIME"

            if exit_price is not None:
                pnl_pu = (exit_price - t["entry"]) if t["dir"]=="long" else (t["entry"] - exit_price)
                trades.append({
                    "entry_time": t["entry_time"], "exit_time": ts, "dir": t["dir"],
                    "entry": t["entry"], "exit": exit_price, "pnl": pnl_pu * t["size"],
                    "r_mult": pnl_pu / t["stop_dist"], "reason": reason,
                    "week": t["entry_time"].isocalendar()[1],
                })
                in_trade = None
            continue

        # ── Entry window check (T-segment) ──
        if not (seg_start <= hour < seg_end):
            continue
        if pd.isna(avg_sp) or spread > 2.0 * avg_sp:
            continue

        # ── Calculate POI and breakout levels ──
        poi_long, poi_short = get_poi(bar, cfg.poi_type)
        if pd.isna(poi_long) or pd.isna(poi_short):
            continue

        if cfg.fract > 0:
            bo_long  = poi_long  + cfg.fract * atr
            bo_short = poi_short - cfg.fract * atr
        else:
            bo_long  = poi_long
            bo_short = poi_short

        # ── Proximity guard: only take the closer breakout ──
        dist_long  = abs(close - bo_long)
        dist_short = abs(close - bo_short)

        can_long  = cfg.direction in ("long", "both")
        can_short = cfg.direction in ("short", "both")

        if cfg.proximity_guard and cfg.direction == "both":
            if dist_long <= dist_short:
                can_short = False
            else:
                can_long = False

        # ── LONG entry ──
        if can_long and close > bo_long and close > bar["sma20"] and bar["sma20"] > bar["sma50"]:
            if check_filter(bar, cfg.filter1, "long"):
                key = (date, "long")
                if key not in session_used:
                    sd = SL_ATR * atr; sz = FIXED_RISK / sd
                    in_trade = {
                        "entry_time":ts, "entry":close, "dir":"long",
                        "sl":close-sd, "tp":close+TP_ATR*atr,
                        "atr":atr, "stop_dist":sd, "size":sz,
                        "be":False, "trailing":False, "mae_price":0.0, "mfe_price":0.0,
                    }
                    session_used[key] = True
                    continue

        # ── SHORT entry ──
        if can_short and close < bo_short and close < bar["sma20"] and bar["sma20"] < bar["sma50"]:
            if check_filter(bar, cfg.filter1, "short"):
                key = (date, "short")
                if key not in session_used:
                    sd = SL_ATR * atr; sz = FIXED_RISK / sd
                    in_trade = {
                        "entry_time":ts, "entry":close, "dir":"short",
                        "sl":close+sd, "tp":close-TP_ATR*atr,
                        "atr":atr, "stop_dist":sd, "size":sz,
                        "be":False, "trailing":False, "mae_price":0.0, "mfe_price":0.0,
                    }
                    session_used[key] = True

    return trades


def summarise(trades):
    if not trades or len(trades) < 3:
        return None
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    pf = gp/gl if gl > 0 else float("inf")
    r_mults = [t["r_mult"] for t in trades]

    week_pnl = {}
    for t in trades:
        w = t["week"]
        week_pnl[w] = week_pnl.get(w, 0) + t["pnl"]
    weeks_pos = sum(1 for v in week_pnl.values() if v > 0)

    mid = len(trades) // 2
    h1 = sum(t["pnl"] for t in trades[:mid])
    h2 = sum(t["pnl"] for t in trades[mid:])

    return {
        "n": len(trades), "wr": len(wins)/len(trades)*100, "pf": pf,
        "net": sum(pnls), "net_pct": sum(pnls)/ACCOUNT_EQUITY*100,
        "avg_r": np.mean(r_mults),
        "weeks_pos": weeks_pos, "weeks_total": len(week_pnl),
        "h1": h1, "h2": h2,
        "robust": h1 > 0 and h2 > 0 and weeks_pos/max(len(week_pnl),1) >= 0.5,
    }


# ─────────────────────────────────────────
# MAIN: SCAN ALL COMBOS
# ─────────────────────────────────────────
def main():
    print("Loading data...")
    data = {}
    for sym in SYMBOLS:
        df = load_and_build(sym)
        if df is not None:
            data[sym] = df
            print(f"  {sym}: {len(df)} bars")

    # Define search space (kept focused to avoid overfit)
    poi_types  = ["asia", "prev_close", "day_open", "minmax_pc_do", "cum_ma", "ema20", "bb"]
    fracts     = [0.0, 0.3, 0.5, 0.8, 1.0]
    filters    = ["none", "vol_expanding", "vol_contracting", "dmi_trend", "dmi_counter",
                  "adx_trending", "adx_ranging", "pullback_atr", "close_vs_open", "close_vs_open_counter"]
    t_segments = [0, 1, 2, 3]
    directions = ["long", "short"]

    # For each symbol, test the BOS parameter space
    all_results = []
    total_combos = len(poi_types) * len(fracts) * len(filters) * len(t_segments) * len(directions)
    print(f"\nScanning {total_combos} configs per symbol, {len(data)} symbols...")
    print(f"Total runs: {total_combos * len(data):,}\n")

    for sym_idx, (sym, df) in enumerate(sorted(data.items())):
        print(f"[{sym_idx+1}/{len(data)}] {sym}...", end=" ", flush=True)
        sym_best = []

        for poi in poi_types:
            for fract in fracts:
                # Skip fract=0 for POIs that need distance (prev_close, ema, bb already have structure)
                if poi in ("prev_close", "day_open", "cum_ma", "ema20") and fract == 0:
                    fract = 0.5  # need some distance from these flat reference levels

                for filt in filters:
                    for tseg in t_segments:
                        for direction in directions:
                            cfg = BOSConfig(
                                direction=direction, poi_type=poi, fract=fract,
                                filter1=filt, t_segment=tseg, proximity_guard=False,
                            )
                            trades = run_bos(df, cfg)
                            s = summarise(trades)
                            if s is None:
                                continue
                            if s["pf"] > 1.0 and s["n"] >= 5:
                                sym_best.append({
                                    "symbol": sym, "direction": direction,
                                    "poi": poi, "fract": fract,
                                    "filter": filt, "tseg": tseg,
                                    **s,
                                })

        # Keep top 5 per symbol by PF (with robustness tiebreak)
        sym_best.sort(key=lambda x: (x["robust"], x["pf"]), reverse=True)
        all_results.extend(sym_best[:5])
        n_profitable = len(sym_best)
        n_robust = sum(1 for r in sym_best if r["robust"])
        print(f"{n_profitable} profitable configs found, {n_robust} robust")

    # ── Print results ──
    print(f"\n{'='*120}")
    print(f"  TOP BOS CONFIGURATIONS (best per symbol, sorted by robustness then PF)")
    print(f"{'='*120}")
    print(f"  {'Symbol':<10} {'Dir':<6} {'POI':<16} {'Fract':>6} {'Filter':<22} {'TSeg':>5}"
          f" {'N':>4} {'WR':>6} {'PF':>6} {'Net%':>8} {'WkP':>5} {'H1$k':>7} {'H2$k':>7} {'Rob':>5}")
    print(f"  {'-'*116}")

    all_results.sort(key=lambda x: (x["robust"], x["pf"]), reverse=True)
    for r in all_results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 10 else "inf"
        rob_s = "YES" if r["robust"] else "no"
        print(
            f"  {r['symbol']:<10} {r['direction']:<6} {r['poi']:<16} {r['fract']:>6.1f}"
            f" {r['filter']:<22} {r['tseg']:>5}"
            f" {r['n']:>4} {r['wr']:>5.0f}% {pf_s:>6} {r['net_pct']:>+7.2f}%"
            f" {r['weeks_pos']}/{r['weeks_total']}"
            f" {r['h1']/1000:>+6.1f} {r['h2']/1000:>+6.1f} {rob_s:>5}"
        )

    # ── Compare: baseline (our original Asia breakout) vs best BOS config per symbol ──
    print(f"\n{'='*120}")
    print(f"  BASELINE vs BEST BOS: per-symbol comparison")
    print(f"{'='*120}")
    print(f"  {'Symbol':<10} {'Dir':<6} | {'Baseline PF':>12} {'Baseline Net%':>14} | {'BOS PF':>8} {'BOS Net%':>10} {'BOS Config'}")
    print(f"  {'-'*116}")

    for sym in sorted(data.keys()):
        # Baseline: original Asia breakout, best direction
        for d in ["long", "short"]:
            base_cfg = BOSConfig(direction=d, poi_type="asia", fract=0.0, filter1="none", t_segment=0)
            base_trades = run_bos(data[sym], base_cfg)
            base_s = summarise(base_trades)

            # Best BOS for this symbol+direction
            best = None
            for r in all_results:
                if r["symbol"] == sym and r["direction"] == d:
                    best = r
                    break

            base_pf  = f"{base_s['pf']:.2f}" if base_s and base_s['pf'] < 10 else ("inf" if base_s else "N/A")
            base_net = f"{base_s['net_pct']:+.2f}%" if base_s else "N/A"

            if best:
                bos_pf  = f"{best['pf']:.2f}" if best['pf'] < 10 else "inf"
                bos_net = f"{best['net_pct']:+.2f}%"
                bos_cfg = f"POI={best['poi']} F={best['fract']:.1f} filt={best['filter']} T={best['tseg']}"
                improved = " <<<" if best and base_s and best["pf"] > base_s.get("pf", 0) else ""
            else:
                bos_pf = bos_net = bos_cfg = "---"
                improved = ""

            print(f"  {sym:<10} {d:<6} | {base_pf:>12} {base_net:>14} | {bos_pf:>8} {bos_net:>10} {bos_cfg}{improved}")

    # ── Save chart: top configs equity-like R-curves ──
    robust_results = [r for r in all_results if r["robust"]][:12]
    if robust_results:
        n_plots = len(robust_results)
        cols = min(4, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, r in enumerate(robust_results):
            ax = axes_flat[idx]
            # Re-run to get trade list
            cfg = BOSConfig(direction=r["direction"], poi_type=r["poi"], fract=r["fract"],
                           filter1=r["filter"], t_segment=r["tseg"], proximity_guard=False)
            trades = run_bos(data[r["symbol"]], cfg)
            cum_r = np.cumsum([t["r_mult"] for t in trades])
            color = "#4CAF50" if cum_r[-1] > 0 else "#F44336"
            ax.plot(cum_r, linewidth=1.5, color=color)
            ax.fill_between(range(len(cum_r)), cum_r, 0, alpha=0.15, color=color)
            ax.axhline(0, color="gray", linewidth=0.5)
            mid = len(cum_r) // 2
            ax.axvline(mid, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
            pf_s = f"{r['pf']:.1f}" if r['pf'] < 10 else "inf"
            ax.set_title(f"{r['symbol']} {r['direction'].upper()}\nPOI={r['poi']} F={r['fract']:.1f}\n"
                        f"filt={r['filter']} T={r['tseg']}\nPF={pf_s} n={r['n']} WR={r['wr']:.0f}%", fontsize=8)
            ax.set_xlabel("Trade #", fontsize=7)
            ax.set_ylabel("Cum R", fontsize=7)

        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle("BOS-Enhanced Robust Edges (R-curves, orange=midpoint)", fontsize=12)
        plt.tight_layout()
        plt.savefig("bos_edges.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("\nSaved: bos_edges.png")


if __name__ == "__main__":
    main()
