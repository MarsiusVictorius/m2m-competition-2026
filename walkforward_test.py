"""
V1.4 Walk-Forward Robustness Test
1. Multiple split ratios: 60/40, 67/33, 75/25, 80/20
2. Rolling walk-forward: sliding 10-day test windows
3. Purged k-fold: 5 time blocks, rotate holdout with 1-day gap
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
FIXED_RISK     = 10_000.0
MAX_POSITIONS  = 6
TARGET_MAX_LEVERAGE = 20.0
SLIPPAGE_SPREAD_MULT = 0.5
SL_ATR = 1.5; TP_ATR = 2.0; BE_ATR = 1.0; TRAIL_ON = 1.0; TRAIL_D = 1.0
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


def run_period(dfs, start_date, end_date):
    """Run portfolio on a specific date range. Returns dict with metrics."""
    all_times = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    all_times = [t for t in all_times if start_date <= t.date() <= end_date]
    if not all_times:
        return {"n": 0, "wr": 0, "pf": 0, "net_pct": 0, "net_dollar": 0, "dd": 0}

    equity = ACCOUNT_EQUITY
    open_trades = []; closed_trades = []; session_entries = {}
    pyramid_groups = {}; group_ctr = 0
    peak_equity = equity

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
                    sl = ep - SL_ATR*atr if d=="long" else ep + SL_ATR*atr
                    tp = ep + TP_ATR*atr if d=="long" else ep - TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
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
                    sl = ep - SL_ATR*atr; tp = ep + TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
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
                    sl = ep + SL_ATR*atr; tp = ep - TP_ATR*atr
                    sz = FIXED_RISK / (SL_ATR * atr)
                    sz = cap_size(sz, ep, open_trades, equity)
                    if sz <= 0: continue
                    group_ctr += 1; gid = f"{cfg.label}_S{group_ctr}"
                    trade = Trade(sym, "short", 1, gid, ts, ep, sl, tp, atr, sz, cfg.label,
                                 slippage_cost=abs(close-ep)*sz)
                    open_trades.append(trade); pyramid_groups[gid] = [trade]
                    session_entries[entry_key] = True

        # Track peak for drawdown
        unrealised = sum(t.unrealised(dfs[t.symbol].loc[ts, "close"])
                        for t in open_trades if t.symbol in dfs and ts in dfs[t.symbol].index)
        current_eq = equity + unrealised
        if current_eq > peak_equity:
            peak_equity = current_eq

    # Close remaining
    for t in list(open_trades):
        t.exit_price = dfs[t.symbol]["close"].iloc[-1]
        t.exit_time = all_times[-1]; t.exit_reason = "EOD"
        t.pnl = (t.exit_price - t.entry_price) * t.size if t.direction=="long" else (t.entry_price - t.exit_price) * t.size
        equity += t.pnl; closed_trades.append(t)

    pnls = [t.pnl for t in closed_trades]
    if not pnls:
        return {"n": 0, "wr": 0, "pf": 0, "net_pct": 0, "net_dollar": 0, "dd": 0}
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0
    pf = gp / gl if gl > 0 else float("inf")
    net_dollar = sum(pnls)
    net_pct = net_dollar / ACCOUNT_EQUITY * 100
    dd = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else 0

    return {"n": len(pnls), "wr": len(wins)/len(pnls)*100, "pf": pf,
            "net_pct": net_pct, "net_dollar": net_dollar, "dd": dd,
            "avg_per_trade": net_dollar / len(pnls)}


def main():
    needed = sorted(set(c.symbol for c in CONFIGS))
    print("Loading data...")
    dfs = {}
    for sym in needed:
        df = load_and_build(sym)
        if df is not None: dfs[sym] = df; print(f"  {sym}: {len(df)} bars")

    all_dates = sorted(set(d for df in dfs.values() for d in df.index.date))
    total_days = len(all_dates)
    print(f"\n  Date range: {all_dates[0]} to {all_dates[-1]} ({total_days} trading days)")

    all_results = []

    # ─────────────────────────────────────────
    # TEST 1: Multiple split ratios
    # ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  TEST 1: MULTIPLE TRAIN/TEST SPLIT RATIOS")
    print(f"{'='*80}")

    splits = [(60, 40), (67, 33), (75, 25), (80, 20)]
    split_results = []

    for train_pct, test_pct in splits:
        cut = int(total_days * train_pct / 100)
        train_end = all_dates[cut - 1]
        test_start = all_dates[cut]

        print(f"\n  Split {train_pct}/{test_pct}: train {all_dates[0]}..{train_end} ({cut}d), test {test_start}..{all_dates[-1]} ({total_days-cut}d)")

        m_train = run_period(dfs, all_dates[0], train_end)
        m_test  = run_period(dfs, test_start, all_dates[-1])

        pf_tr = f"{m_train['pf']:.2f}" if m_train['pf'] < 100 else "inf"
        pf_te = f"{m_test['pf']:.2f}" if m_test['pf'] < 100 else "inf"
        print(f"    TRAIN: N={m_train['n']:>3}, WR={m_train['wr']:>5.1f}%, PF={pf_tr:>6}, Net={m_train['net_pct']:>+6.1f}%")
        print(f"    TEST:  N={m_test['n']:>3}, WR={m_test['wr']:>5.1f}%, PF={pf_te:>6}, Net={m_test['net_pct']:>+6.1f}%")

        split_results.append({
            "split": f"{train_pct}/{test_pct}",
            "train_days": cut, "test_days": total_days - cut,
            "train": m_train, "test": m_test
        })

    # ─────────────────────────────────────────
    # TEST 2: Rolling walk-forward (10-day test windows)
    # ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  TEST 2: ROLLING WALK-FORWARD (10-day sliding test window)")
    print(f"{'='*80}")

    window_size = 10
    rolling_results = []
    step = 0

    for start_idx in range(0, total_days - window_size + 1, window_size):
        end_idx = min(start_idx + window_size - 1, total_days - 1)
        w_start = all_dates[start_idx]
        w_end = all_dates[end_idx]
        step += 1

        print(f"\n  Window {step}: {w_start} to {w_end} ({end_idx - start_idx + 1} days)")
        m = run_period(dfs, w_start, w_end)
        pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
        print(f"    N={m['n']:>3}, WR={m['wr']:>5.1f}%, PF={pf_s:>6}, Net={m['net_pct']:>+6.1f}%, $/trade={m.get('avg_per_trade',0):>+,.0f}")
        rolling_results.append({"window": step, "start": w_start, "end": w_end, **m})

    # ─────────────────────────────────────────
    # TEST 3: Purged 5-fold cross-validation
    # ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  TEST 3: PURGED 5-FOLD CROSS-VALIDATION (1-day purge gap)")
    print(f"{'='*80}")

    n_folds = 5
    fold_size = total_days // n_folds
    kfold_results = []

    for fold in range(n_folds):
        test_start_idx = fold * fold_size
        test_end_idx = min((fold + 1) * fold_size - 1, total_days - 1)
        if fold == n_folds - 1:
            test_end_idx = total_days - 1

        test_start_d = all_dates[test_start_idx]
        test_end_d = all_dates[test_end_idx]

        print(f"\n  Fold {fold+1}: holdout = {test_start_d} to {test_end_d} ({test_end_idx - test_start_idx + 1} days)")
        m = run_period(dfs, test_start_d, test_end_d)
        pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
        print(f"    N={m['n']:>3}, WR={m['wr']:>5.1f}%, PF={pf_s:>6}, Net={m['net_pct']:>+6.1f}%, $/trade={m.get('avg_per_trade',0):>+,.0f}")
        kfold_results.append({"fold": fold+1, "start": test_start_d, "end": test_end_d, **m})

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*80}")

    print(f"\n  1) SPLIT RATIOS (test periods):")
    print(f"  {'Split':<10} {'N':>5} {'WR':>6} {'PF':>7} {'Net%':>8} {'$/trade':>10}")
    print(f"  {'-'*47}")
    for s in split_results:
        m = s["test"]
        pf_s = f"{m['pf']:.2f}" if m['pf'] < 100 else "inf"
        print(f"  {s['split']:<10} {m['n']:>5} {m['wr']:>5.0f}% {pf_s:>7} {m['net_pct']:>+7.1f}% ${m.get('avg_per_trade',0):>+8,.0f}")

    all_test_profitable = all(s["test"]["net_pct"] > 0 for s in split_results)
    print(f"\n  All test periods profitable: {'YES' if all_test_profitable else 'NO'}")

    print(f"\n  2) ROLLING WINDOWS:")
    print(f"  {'Window':<10} {'Dates':<25} {'N':>5} {'WR':>6} {'PF':>7} {'Net%':>8}")
    print(f"  {'-'*62}")
    for r in rolling_results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        print(f"  W{r['window']:<9} {str(r['start'])+'..'+str(r['end']):<25} {r['n']:>5} {r['wr']:>5.0f}% {pf_s:>7} {r['net_pct']:>+7.1f}%")

    all_rolling_profitable = all(r["net_pct"] > 0 for r in rolling_results)
    rolling_nets = [r["net_pct"] for r in rolling_results]
    print(f"\n  All windows profitable: {'YES' if all_rolling_profitable else 'NO'}")
    print(f"  Min window return: {min(rolling_nets):+.1f}%, Max: {max(rolling_nets):+.1f}%, Avg: {np.mean(rolling_nets):+.1f}%")

    print(f"\n  3) PURGED K-FOLD:")
    print(f"  {'Fold':<8} {'Dates':<25} {'N':>5} {'WR':>6} {'PF':>7} {'Net%':>8}")
    print(f"  {'-'*60}")
    for r in kfold_results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
        print(f"  F{r['fold']:<7} {str(r['start'])+'..'+str(r['end']):<25} {r['n']:>5} {r['wr']:>5.0f}% {pf_s:>7} {r['net_pct']:>+7.1f}%")

    all_folds_profitable = all(r["net_pct"] > 0 for r in kfold_results)
    fold_nets = [r["net_pct"] for r in kfold_results]
    print(f"\n  All folds profitable: {'YES' if all_folds_profitable else 'NO'}")
    print(f"  Min fold return: {min(fold_nets):+.1f}%, Max: {max(fold_nets):+.1f}%, Avg: {np.mean(fold_nets):+.1f}%")

    # VERDICT
    total_tests = len(split_results) + len(rolling_results) + len(kfold_results)
    profitable_tests = (sum(1 for s in split_results if s["test"]["net_pct"] > 0) +
                       sum(1 for r in rolling_results if r["net_pct"] > 0) +
                       sum(1 for r in kfold_results if r["net_pct"] > 0))
    all_pf_above_1 = (all(s["test"]["pf"] > 1 for s in split_results) and
                      all(r["pf"] > 1 for r in rolling_results) and
                      all(r["pf"] > 1 for r in kfold_results))

    print(f"\n  {'='*50}")
    verdict = "ROBUST" if profitable_tests == total_tests and all_pf_above_1 else \
              "MODERATE" if profitable_tests >= total_tests * 0.8 else "FRAGILE"
    print(f"  VERDICT: {verdict}")
    print(f"  Profitable periods: {profitable_tests}/{total_tests}")
    print(f"  All PF > 1.0: {'YES' if all_pf_above_1 else 'NO'}")
    print(f"  {'='*50}")

    # ─────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

    # 1) Split ratios: train vs test bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(split_results))
    w = 0.35
    train_nets = [s["train"]["net_pct"] for s in split_results]
    test_nets = [s["test"]["net_pct"] for s in split_results]
    ax1.bar(x - w/2, train_nets, w, label="Train", color="#2196F3", alpha=0.8)
    ax1.bar(x + w/2, test_nets, w, label="Test (OOS)", color="#FF9800", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s["split"] for s in split_results])
    ax1.set_ylabel("Net Return %")
    ax1.set_title("Split Ratio: Train vs Test Return")
    ax1.legend()
    ax1.axhline(0, color="gray", linewidth=0.5)
    for i, (tr, te) in enumerate(zip(train_nets, test_nets)):
        ax1.text(i - w/2, tr + 0.3, f"{tr:+.1f}%", ha="center", fontsize=8)
        ax1.text(i + w/2, te + 0.3, f"{te:+.1f}%", ha="center", fontsize=8)

    # 2) Split ratios: PF comparison
    ax2 = fig.add_subplot(gs[0, 1])
    train_pfs = [min(s["train"]["pf"], 10) for s in split_results]
    test_pfs = [min(s["test"]["pf"], 10) for s in split_results]
    ax2.bar(x - w/2, train_pfs, w, label="Train", color="#2196F3", alpha=0.8)
    ax2.bar(x + w/2, test_pfs, w, label="Test (OOS)", color="#FF9800", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s["split"] for s in split_results])
    ax2.set_ylabel("Profit Factor (capped 10)")
    ax2.set_title("Split Ratio: Train vs Test PF")
    ax2.legend()
    ax2.axhline(1.0, color="red", linewidth=0.8, linestyle="--", label="Break-even")

    # 3) Rolling windows bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    rx = np.arange(len(rolling_results))
    r_nets = [r["net_pct"] for r in rolling_results]
    r_colors = ["#4CAF50" if n > 0 else "#F44336" for n in r_nets]
    ax3.bar(rx, r_nets, color=r_colors, alpha=0.8)
    ax3.set_xticks(rx)
    ax3.set_xticklabels([f"W{r['window']}\n{r['start']}\n{r['end']}" for r in rolling_results], fontsize=7)
    ax3.set_ylabel("Net Return %")
    ax3.set_title(f"Rolling Walk-Forward ({window_size}-day windows)")
    ax3.axhline(0, color="gray", linewidth=0.8)
    for i, n in enumerate(r_nets):
        ax3.text(i, n + 0.2, f"{n:+.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 4) K-fold bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    kx = np.arange(len(kfold_results))
    k_nets = [r["net_pct"] for r in kfold_results]
    k_colors = ["#4CAF50" if n > 0 else "#F44336" for n in k_nets]
    ax4.bar(kx, k_nets, color=k_colors, alpha=0.8)
    ax4.set_xticks(kx)
    ax4.set_xticklabels([f"F{r['fold']}\n{r['start']}\n{r['end']}" for r in kfold_results], fontsize=7)
    ax4.set_ylabel("Net Return %")
    ax4.set_title("Purged 5-Fold Cross-Validation")
    ax4.axhline(0, color="gray", linewidth=0.8)
    for i, n in enumerate(k_nets):
        ax4.text(i, n + 0.2, f"{n:+.1f}%", ha="center", fontsize=9, fontweight="bold")

    # 5) Win rate consistency across all tests
    ax5 = fig.add_subplot(gs[1, 1])
    all_wrs = ([s["test"]["wr"] for s in split_results] +
               [r["wr"] for r in rolling_results] +
               [r["wr"] for r in kfold_results])
    all_labels = ([f"Split {s['split']}" for s in split_results] +
                  [f"Roll W{r['window']}" for r in rolling_results] +
                  [f"Fold {r['fold']}" for r in kfold_results])
    wr_colors = ["#4CAF50" if w >= 50 else "#F44336" for w in all_wrs]
    ax5.barh(all_labels, all_wrs, color=wr_colors, alpha=0.8)
    ax5.axvline(50, color="red", linestyle="--", linewidth=0.8)
    ax5.set_xlabel("Win Rate %")
    ax5.set_title("Win Rate Across All Tests")
    ax5.set_xlim(0, 100)
    ax5.tick_params(axis="y", labelsize=8)

    # 6) Verdict text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    bg = "#e8f5e9" if verdict == "ROBUST" else "#fff3e0" if verdict == "MODERATE" else "#ffebee"
    verdict_text = (
        f"WALK-FORWARD VERDICT: {verdict}\n"
        f"{'='*40}\n\n"
        f"Split ratio tests:    {sum(1 for s in split_results if s['test']['net_pct']>0)}/{len(split_results)} profitable\n"
        f"Rolling windows:      {sum(1 for r in rolling_results if r['net_pct']>0)}/{len(rolling_results)} profitable\n"
        f"K-fold holdouts:      {sum(1 for r in kfold_results if r['net_pct']>0)}/{len(kfold_results)} profitable\n"
        f"{'─'*40}\n"
        f"Total:                {profitable_tests}/{total_tests} profitable\n"
        f"All PF > 1.0:         {'YES' if all_pf_above_1 else 'NO'}\n\n"
        f"Split test returns:   {min(test_nets):+.1f}% to {max(test_nets):+.1f}%\n"
        f"Rolling returns:      {min(rolling_nets):+.1f}% to {max(rolling_nets):+.1f}%\n"
        f"Fold returns:         {min(fold_nets):+.1f}% to {max(fold_nets):+.1f}%\n\n"
        f"Avg return/test:      {np.mean(test_nets + rolling_nets + fold_nets):+.1f}%\n"
        f"Min return/test:      {min(test_nets + rolling_nets + fold_nets):+.1f}%\n"
        f"Consistency:          {profitable_tests/total_tests*100:.0f}%"
    )
    ax6.text(0.05, 0.95, verdict_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=bg, alpha=0.9))

    plt.suptitle("V1.4 Walk-Forward Robustness Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("walkforward_v1.4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: walkforward_v1.4.png")


if __name__ == "__main__":
    main()
