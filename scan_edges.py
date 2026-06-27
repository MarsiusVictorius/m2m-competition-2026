"""
Edge scanner: run every symbol in long-only / short-only / both
with FIXED position sizing ($10k risk per trade, no dynamic scaling).
Finds which symbol+direction combos actually have edge.
"""

import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR       = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
CANDLE_MINUTES = 15
ACCOUNT_EQUITY = 1_000_000.0
FIXED_RISK     = 10_000.0   # $10k per trade, no compounding

# Simplified strategy params
SL_ATR   = 1.5
TP_ATR   = 2.0
BE_ATR   = 1.0      # move to breakeven after 1 ATR profit
TRAIL_ON = 1.0      # trail activates after 1 ATR profit
TRAIL_D  = 1.0      # trail distance in ATR

ALL_SYMBOLS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD",
    "GBPUSD", "NZDUSD", "UKOILUSD", "USDCAD", "USDCHF", "USDCNH", "USDHKD",
    "USDJPY", "USOILUSD", "XAGUSD", "XAUCNH", "XAUGCNH", "XAUHKD", "XAUKUSD", "XAUUSD",
]


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

    ohlc["sma20"] = ohlc["close"].rolling(20).mean()
    ohlc["sma50"] = ohlc["close"].rolling(50).mean()
    prev = ohlc["close"].shift(1)
    tr = pd.concat([ohlc["high"]-ohlc["low"], (ohlc["high"]-prev).abs(), (ohlc["low"]-prev).abs()], axis=1).max(axis=1)
    ohlc["atr14"]  = tr.rolling(14).mean()
    ohlc["_date"]  = ohlc.index.date
    ohlc["_hour"]  = ohlc.index.hour
    asia = ohlc[ohlc["_hour"] < 7].groupby("_date").agg(asia_high=("high","max"), asia_low=("low","min"))
    ohlc = ohlc.join(asia, on="_date")
    ohlc["avg_spread"] = ohlc["spread"].rolling(20).mean()
    ohlc.drop(columns=["_date", "_hour"], inplace=True)
    return ohlc


def run_simple(df, direction="both"):
    """
    Minimal backtest: fixed $10k risk, no compounding, no pyramiding.
    Returns list of trade dicts.
    """
    trades = []
    in_trade = None
    session_used = {}  # (date, dir) -> True

    for ts, bar in df.iterrows():
        hour = ts.hour
        date = ts.date()

        close = bar["close"]
        high  = bar["high"]
        low   = bar["low"]
        atr   = bar["atr14"]
        sma20 = bar["sma20"]
        sma50 = bar["sma50"]
        ah    = bar["asia_high"]
        al    = bar["asia_low"]
        spread = bar["spread"]
        avg_sp = bar["avg_spread"]

        if pd.isna(atr) or pd.isna(sma20) or pd.isna(sma50) or pd.isna(ah) or pd.isna(al):
            continue

        # ── Manage open trade ──
        if in_trade is not None:
            t = in_trade
            profit_atr = (close - t["entry"]) / t["atr"] if t["dir"] == "long" else (t["entry"] - close) / t["atr"]

            # Breakeven
            if not t["be"] and profit_atr >= BE_ATR:
                t["sl"] = t["entry"]
                t["be"] = True

            # Trailing
            if not t["trailing"] and profit_atr >= TRAIL_ON:
                t["trailing"] = True
            if t["trailing"]:
                if t["dir"] == "long":
                    t["sl"] = max(t["sl"], close - TRAIL_D * atr)
                else:
                    t["sl"] = min(t["sl"], close + TRAIL_D * atr)

            # Track MAE/MFE in price terms
            if t["dir"] == "long":
                upnl = close - t["entry"]
            else:
                upnl = t["entry"] - close
            t["mae_price"] = min(t["mae_price"], upnl)
            t["mfe_price"] = max(t["mfe_price"], upnl)

            # Check exits
            hit_sl = (t["dir"] == "long" and low <= t["sl"]) or (t["dir"] == "short" and high >= t["sl"])
            hit_tp = (t["dir"] == "long" and high >= t["tp"]) or (t["dir"] == "short" and low <= t["tp"])
            time_exit = (hour == 21)

            exit_price = None
            reason = ""
            if hit_tp:
                exit_price = t["tp"]
                reason = "TP"
            elif hit_sl:
                exit_price = t["sl"]
                reason = "SL"
            elif time_exit:
                exit_price = close
                reason = "TIME"

            if exit_price is not None:
                if t["dir"] == "long":
                    pnl_per_unit = exit_price - t["entry"]
                else:
                    pnl_per_unit = t["entry"] - exit_price

                pnl_dollar = pnl_per_unit * t["size"]
                # R-multiple: how many R did this trade make?
                r_mult = pnl_per_unit / t["stop_dist"]

                trades.append({
                    "entry_time": t["entry_time"],
                    "exit_time": ts,
                    "dir": t["dir"],
                    "entry": t["entry"],
                    "exit": exit_price,
                    "pnl": pnl_dollar,
                    "r_mult": r_mult,
                    "mae_price": t["mae_price"],
                    "mfe_price": t["mfe_price"],
                    "reason": reason,
                    "atr": t["atr"],
                    "stop_dist": t["stop_dist"],
                })
                in_trade = None
            continue

        # ── Entry logic (07:00-16:00 UTC) ──
        if not (7 <= hour < 16):
            continue
        if pd.isna(avg_sp) or spread > 2.0 * avg_sp:
            continue

        # LONG
        if direction in ("long", "both") and close > ah and close > sma20 and sma20 > sma50:
            key = (date, "long")
            if key not in session_used:
                stop_dist = SL_ATR * atr
                size = FIXED_RISK / stop_dist
                in_trade = {
                    "entry_time": ts, "entry": close, "dir": "long",
                    "sl": close - stop_dist, "tp": close + TP_ATR * atr,
                    "atr": atr, "stop_dist": stop_dist, "size": size,
                    "be": False, "trailing": False,
                    "mae_price": 0.0, "mfe_price": 0.0,
                }
                session_used[key] = True
                continue

        # SHORT
        if direction in ("short", "both") and close < al and close < sma20 and sma20 < sma50:
            key = (date, "short")
            if key not in session_used:
                stop_dist = SL_ATR * atr
                size = FIXED_RISK / stop_dist
                in_trade = {
                    "entry_time": ts, "entry": close, "dir": "short",
                    "sl": close + stop_dist, "tp": close - TP_ATR * atr,
                    "atr": atr, "stop_dist": stop_dist, "size": size,
                    "be": False, "trailing": False,
                    "mae_price": 0.0, "mfe_price": 0.0,
                }
                session_used[key] = True

    return trades


def summarise(trades):
    if not trades:
        return None
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    pf = gp / gl if gl > 0 else float("inf")
    r_mults = [t["r_mult"] for t in trades]
    avg_r = np.mean(r_mults)
    expectancy_r = avg_r  # average R per trade

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    return {
        "n": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pf": pf,
        "net": sum(pnls),
        "net_pct": sum(pnls) / ACCOUNT_EQUITY * 100,
        "avg_r": avg_r,
        "exp_r": expectancy_r,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "reasons": reasons,
    }


def main():
    print("Loading all symbols...\n")
    data = {}
    for sym in ALL_SYMBOLS:
        df = load_and_build(sym)
        if df is not None:
            data[sym] = df
            print(f"  {sym}: {len(df)} candles")

    print(f"\n{'='*100}")
    print(f"  EDGE SCAN: every symbol x {{long, short, both}} -- FIXED $10k risk, no compounding")
    print(f"{'='*100}")
    print(f"  {'Symbol':<12} {'Dir':<7} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'AvgR':>7} {'Net$':>12} {'Net%':>8}  {'TP':>4} {'SL':>4} {'TIME':>5}")
    print(f"  {'-'*96}")

    results = []

    for sym in sorted(data.keys()):
        df = data[sym]
        for d in ["long", "short", "both"]:
            trades = run_simple(df, direction=d)
            s = summarise(trades)
            if s is None:
                continue
            pf_str = f"{s['pf']:.2f}" if s["pf"] < 100 else "inf"
            tp_n = s["reasons"].get("TP", 0)
            sl_n = s["reasons"].get("SL", 0)
            tm_n = s["reasons"].get("TIME", 0)

            flag = ""
            if s["pf"] > 1.0 and s["n"] >= 5:
                flag = " ***"
            elif s["pf"] > 0.9 and s["n"] >= 5:
                flag = " *"

            print(
                f"  {sym:<12} {d:<7} {s['n']:>7} {s['wr']:>7.1f}% {pf_str:>7}"
                f" {s['avg_r']:>+6.2f}R {s['net']:>+11,.0f} {s['net_pct']:>+7.2f}%"
                f"  {tp_n:>4} {sl_n:>4} {tm_n:>5}{flag}"
            )
            results.append({"symbol": sym, "dir": d, **s})

    # ── Top edges ──
    profitable = [r for r in results if r["pf"] > 1.0 and r["n"] >= 5]
    profitable.sort(key=lambda x: x["pf"], reverse=True)

    print(f"\n{'='*100}")
    print(f"  TOP EDGES (PF > 1.0 and >= 5 trades)")
    print(f"{'='*100}")
    if not profitable:
        print("  No profitable edges found.")
    else:
        for r in profitable[:15]:
            pf_str = f"{r['pf']:.2f}" if r["pf"] < 100 else "inf"
            print(
                f"  {r['symbol']:<12} {r['dir']:<7} {r['n']:>3} trades  "
                f"WR {r['wr']:.0f}%  PF {pf_str}  AvgR {r['avg_r']:+.2f}  "
                f"Net {r['net_pct']:+.2f}%"
            )

    # ── Worst performers ──
    losers = [r for r in results if r["n"] >= 5]
    losers.sort(key=lambda x: x["pf"])
    print(f"\n  WORST PERFORMERS (avoid these):")
    for r in losers[:10]:
        pf_str = f"{r['pf']:.2f}"
        print(
            f"  {r['symbol']:<12} {r['dir']:<7} {r['n']:>3} trades  "
            f"WR {r['wr']:.0f}%  PF {pf_str}  AvgR {r['avg_r']:+.2f}  "
            f"Net {r['net_pct']:+.2f}%"
        )

    # ── R-distribution analysis ──
    print(f"\n{'='*100}")
    print(f"  EXIT REASON ANALYSIS (all trades, all symbols)")
    print(f"{'='*100}")
    all_trades = []
    for sym in data:
        all_trades.extend(run_simple(data[sym], "both"))

    if all_trades:
        tp_trades  = [t for t in all_trades if t["reason"] == "TP"]
        sl_trades  = [t for t in all_trades if t["reason"] == "SL"]
        tm_trades  = [t for t in all_trades if t["reason"] == "TIME"]
        print(f"  TP exits:   {len(tp_trades):>4}  avg R: {np.mean([t['r_mult'] for t in tp_trades]):+.2f}" if tp_trades else "  TP exits:      0")
        print(f"  SL exits:   {len(sl_trades):>4}  avg R: {np.mean([t['r_mult'] for t in sl_trades]):+.2f}" if sl_trades else "  SL exits:      0")
        print(f"  TIME exits: {len(tm_trades):>4}  avg R: {np.mean([t['r_mult'] for t in tm_trades]):+.2f}" if tm_trades else "  TIME exits:    0")

        # How many TIME exits were winners vs losers?
        if tm_trades:
            tm_win  = sum(1 for t in tm_trades if t["pnl"] > 0)
            tm_loss = sum(1 for t in tm_trades if t["pnl"] <= 0)
            print(f"    TIME wins: {tm_win}  TIME losses: {tm_loss}")
            print(f"    TIME avg MFE (ATR): {np.mean([t['mfe_price']/t['atr'] for t in tm_trades]):.2f}")
            print(f"    TIME avg MAE (ATR): {np.mean([t['mae_price']/t['atr'] for t in tm_trades]):.2f}")


if __name__ == "__main__":
    main()
