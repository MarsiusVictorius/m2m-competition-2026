"""
Random-search optimizer for the session-breakout strategy.

Parameters searched:
  SL_ATR_MULT  : 0.8 – 2.5
  TP_ATR_MULT  : 1.5 – 5.0
  SMA fast period : 10 – 30  (replaces the fixed sma20 trend filter)
  Entry window end : 14 – 19 UTC

Walk-forward split: train = first 20 calendar days, validate = last 10.

Competition score: 0.7*return + 0.15*(1/abs_drawdown) + 0.10*sharpe
"""

import os
import sys
import copy
import datetime
import random

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtester import (
    load_tick_files, build_ohlc, add_indicators,
    run_portfolio, compute_metrics,
    EDGE_CONFIGS, StrategyConfig, ACCOUNT_EQUITY,
)

# ─────────────────────────────────────────────
# SEARCH CONFIG
# ─────────────────────────────────────────────
N_TRIALS    = 200
RANDOM_SEED = 42

PARAM_SPACE = {
    "sl_atr_mult": (0.8,  2.5),
    "tp_atr_mult": (1.5,  5.0),
    "sma_fast":    (10,   30),   # integer
    "entry_end":   (14,   19),   # integer UTC hour
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def competition_score(metrics: dict) -> float:
    if not metrics or metrics.get("total_trades", 0) == 0:
        return -999.0
    ret = metrics["net_return_pct"]
    dd  = abs(metrics["max_drawdown_pct"])
    sh  = metrics["sharpe_15min"]
    dd_term = 1.0 / dd if dd > 0.01 else 100.0
    return 0.7 * ret + 0.15 * dd_term + 0.10 * sh


def apply_params(base_configs: dict, params: dict) -> dict:
    """Deep-copy configs and apply the trial's param overrides."""
    out = {}
    for sym, cfg in base_configs.items():
        c = copy.copy(cfg)
        c.sl_atr_mult = params["sl_atr_mult"]
        c.tp_atr_mult = params["tp_atr_mult"]
        c.entry_end   = params["entry_end"]
        out[sym] = c
    return out


def inject_fast_sma(dfs: dict, fast_period: int) -> dict:
    """Return new dfs with sma20 column replaced by a rolling mean of fast_period."""
    out = {}
    for sym, df in dfs.items():
        d = df.copy()
        d["sma20"] = d["close"].rolling(fast_period).mean()
        out[sym] = d
    return out


def slice_by_dates(dfs: dict, start: datetime.date, end_excl: datetime.date) -> dict:
    """Return each df filtered to [start, end_excl)."""
    out = {}
    for sym, df in dfs.items():
        mask = (df.index.date >= start) & (df.index.date < end_excl)
        sliced = df[mask]
        if len(sliced) > 0:
            out[sym] = sliced
    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Load and build OHLC once ──
    needed_symbols = sorted(set(cfg.symbol for cfg in EDGE_CONFIGS.values()))
    print(f"Loading data for: {needed_symbols}")
    dfs_base: dict[str, pd.DataFrame] = {}
    for symbol in needed_symbols:
        try:
            ticks = load_tick_files(symbol)
        except FileNotFoundError:
            print(f"  [{symbol}] no data -- skipping")
            continue
        ohlc = build_ohlc(ticks)
        ohlc = add_indicators(ohlc)
        dfs_base[symbol] = ohlc
        print(f"  [{symbol}] {len(ohlc)} candles")

    if not dfs_base:
        print("No data loaded.")
        return

    # ── Derive date range ──
    all_dates = sorted({d for df in dfs_base.values() for d in df.index.date})
    print(f"\nDate range: {all_dates[0]} -> {all_dates[-1]}  ({len(all_dates)} calendar days)")

    if len(all_dates) < 25:
        print(f"Need at least 25 days (20 train + 5 validate); only {len(all_dates)} available.")
        return

    train_dates = all_dates[:20]
    val_dates   = all_dates[20:]          # everything after the training window
    train_start = train_dates[0]
    train_end   = train_dates[-1]
    val_start   = val_dates[0]
    val_end     = val_dates[-1]

    train_end_excl = train_end + datetime.timedelta(days=1)
    val_end_excl   = val_end   + datetime.timedelta(days=1)

    print(f"Train:    {train_start}  ->  {train_end}  ({len(train_dates)} days)")
    print(f"Validate: {val_start}  ->  {val_end}  ({len(val_dates)} days)")

    # ── Build base configs (same dedup logic as backtester.main) ──
    base_configs: dict[str, StrategyConfig] = {}
    for cfg in EDGE_CONFIGS.values():
        if cfg.symbol in dfs_base and cfg.symbol not in base_configs:
            base_configs[cfg.symbol] = cfg

    # ── Random search ──
    rng = random.Random(RANDOM_SEED)
    results = []

    print(f"\nRunning {N_TRIALS} random trials...\n")
    for trial in range(N_TRIALS):
        params = {
            "sl_atr_mult": rng.uniform(*PARAM_SPACE["sl_atr_mult"]),
            "tp_atr_mult": rng.uniform(*PARAM_SPACE["tp_atr_mult"]),
            "sma_fast":    rng.randint(*PARAM_SPACE["sma_fast"]),
            "entry_end":   rng.randint(*PARAM_SPACE["entry_end"]),
        }

        # Inject parameterised fast SMA
        dfs_trial = inject_fast_sma(dfs_base, params["sma_fast"])

        # Split
        dfs_train = slice_by_dates(dfs_trial, train_start, train_end_excl)
        dfs_val   = slice_by_dates(dfs_trial, val_start,   val_end_excl)

        configs = apply_params(base_configs, params)
        train_cfgs = {s: c for s, c in configs.items() if s in dfs_train}
        val_cfgs   = {s: c for s, c in configs.items() if s in dfs_val}

        try:
            tr_trades, tr_eq = run_portfolio(dfs_train, train_cfgs)
            tr_m  = compute_metrics(tr_trades, tr_eq)
            tr_sc = competition_score(tr_m)

            v_trades, v_eq = run_portfolio(dfs_val, val_cfgs)
            v_m   = compute_metrics(v_trades, v_eq)
            v_sc  = competition_score(v_m)
        except Exception as exc:
            print(f"  Trial {trial + 1} error: {exc}")
            continue

        results.append({
            "trial":      trial + 1,
            "sl_mult":    round(params["sl_atr_mult"], 3),
            "tp_mult":    round(params["tp_atr_mult"], 3),
            "sma_fast":   params["sma_fast"],
            "entry_end":  params["entry_end"],
            # In-sample
            "is_trades":  tr_m.get("total_trades",    0),
            "is_return":  round(tr_m.get("net_return_pct",   0.0), 3),
            "is_dd":      round(tr_m.get("max_drawdown_pct", 0.0), 3),
            "is_sharpe":  round(tr_m.get("sharpe_15min",     0.0), 4),
            "is_score":   round(tr_sc, 4),
            # Out-of-sample
            "oos_trades": v_m.get("total_trades",    0),
            "oos_return": round(v_m.get("net_return_pct",   0.0), 3),
            "oos_dd":     round(v_m.get("max_drawdown_pct", 0.0), 3),
            "oos_sharpe": round(v_m.get("sharpe_15min",     0.0), 4),
            "oos_score":  round(v_sc, 4),
        })

        if (trial + 1) % 25 == 0:
            print(f"  {trial + 1}/{N_TRIALS} trials done")

    if not results:
        print("No valid results generated.")
        return

    df_all = pd.DataFrame(results)
    df_top = (
        df_all
        .sort_values("oos_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    df_top.index += 1  # rank starts at 1

    # ── Print ranked table ──
    COL_W = 110
    print("\n" + "=" * COL_W)
    print("  TOP 20 PARAMETER SETS  (ranked by out-of-sample competition score)")
    print("=" * COL_W)
    print(
        f"  {'Rk':>2}  {'SL':>5}  {'TP':>5}  {'SMAf':>4}  {'End':>3}"
        f"  {'IS_Tr':>5}  {'IS_Ret%':>8}  {'IS_DD%':>7}  {'IS_Sh':>7}  {'IS_Sc':>7}"
        f"  {'OS_Tr':>5}  {'OS_Ret%':>8}  {'OS_DD%':>7}  {'OS_Sh':>7}  {'OS_Sc':>7}"
    )
    print("  " + "-" * (COL_W - 2))
    for rank, row in df_top.iterrows():
        print(
            f"  {rank:>2}  {row.sl_mult:>5.2f}  {row.tp_mult:>5.2f}"
            f"  {row.sma_fast:>4}  {row.entry_end:>3}"
            f"  {int(row.is_trades):>5}  {row.is_return:>+8.2f}  {row.is_dd:>7.2f}"
            f"  {row.is_sharpe:>7.4f}  {row.is_score:>7.3f}"
            f"  {int(row.oos_trades):>5}  {row.oos_return:>+8.2f}  {row.oos_dd:>7.2f}"
            f"  {row.oos_sharpe:>7.4f}  {row.oos_score:>7.3f}"
        )
    print("=" * COL_W)

    # ── Save full results ──
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimizer_results.csv")
    df_all.sort_values("oos_score", ascending=False).to_csv(out_path, index=False)
    print(f"\nFull {len(results)}-trial results saved: {out_path}")

    # ── Print best overall params ──
    best = df_top.iloc[0]
    print(f"\nBest params (OOS score {best.oos_score:.3f}):")
    print(f"  SL mult   = {best.sl_mult}")
    print(f"  TP mult   = {best.tp_mult}")
    print(f"  SMA fast  = {int(best.sma_fast)}")
    print(f"  Entry end = {int(best.entry_end)}:00 UTC")


if __name__ == "__main__":
    main()
