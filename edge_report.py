"""
Full edge report: every symbol x {long, short, both}
Produces a comprehensive comparison chart + overfitting analysis.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

DATA_DIR       = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
CANDLE_MINUTES = 15
ACCOUNT_EQUITY = 1_000_000.0
FIXED_RISK     = 10_000.0

SL_ATR = 1.5
TP_ATR = 2.0
BE_ATR = 1.0
TRAIL_ON = 1.0
TRAIL_D  = 1.0

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
    trades = []
    in_trade = None
    session_used = {}
    for ts, bar in df.iterrows():
        hour = ts.hour
        date = ts.date()
        close, high, low = bar["close"], bar["high"], bar["low"]
        atr, sma20, sma50 = bar["atr14"], bar["sma20"], bar["sma50"]
        ah, al = bar["asia_high"], bar["asia_low"]
        spread, avg_sp = bar["spread"], bar["avg_spread"]
        if pd.isna(atr) or pd.isna(sma20) or pd.isna(sma50) or pd.isna(ah) or pd.isna(al):
            continue
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
            if t["dir"] == "long": upnl = close - t["entry"]
            else:                  upnl = t["entry"] - close
            t["mae_price"] = min(t["mae_price"], upnl)
            t["mfe_price"] = max(t["mfe_price"], upnl)
            hit_sl = (t["dir"]=="long" and low<=t["sl"]) or (t["dir"]=="short" and high>=t["sl"])
            hit_tp = (t["dir"]=="long" and high>=t["tp"]) or (t["dir"]=="short" and low<=t["tp"])
            time_exit = (hour == 21)
            exit_price = reason = None
            if hit_tp:    exit_price, reason = t["tp"], "TP"
            elif hit_sl:  exit_price, reason = t["sl"], "SL"
            elif time_exit: exit_price, reason = close, "TIME"
            if exit_price is not None:
                pnl_pu = (exit_price - t["entry"]) if t["dir"]=="long" else (t["entry"] - exit_price)
                trades.append({
                    "entry_time": t["entry_time"], "exit_time": ts, "dir": t["dir"],
                    "entry": t["entry"], "exit": exit_price, "pnl": pnl_pu * t["size"],
                    "r_mult": pnl_pu / t["stop_dist"], "reason": reason, "atr": t["atr"],
                    "week": t["entry_time"].isocalendar()[1],
                })
                in_trade = None
            continue
        if not (7 <= hour < 16): continue
        if pd.isna(avg_sp) or spread > 2.0 * avg_sp: continue
        if direction in ("long","both") and close > ah and close > sma20 and sma20 > sma50:
            key = (date, "long")
            if key not in session_used:
                sd = SL_ATR * atr; sz = FIXED_RISK / sd
                in_trade = {"entry_time":ts,"entry":close,"dir":"long","sl":close-sd,"tp":close+TP_ATR*atr,
                            "atr":atr,"stop_dist":sd,"size":sz,"be":False,"trailing":False,"mae_price":0.0,"mfe_price":0.0}
                session_used[key] = True; continue
        if direction in ("short","both") and close < al and close < sma20 and sma20 < sma50:
            key = (date, "short")
            if key not in session_used:
                sd = SL_ATR * atr; sz = FIXED_RISK / sd
                in_trade = {"entry_time":ts,"entry":close,"dir":"short","sl":close+sd,"tp":close-TP_ATR*atr,
                            "atr":atr,"stop_dist":sd,"size":sz,"be":False,"trailing":False,"mae_price":0.0,"mfe_price":0.0}
                session_used[key] = True
    return trades


def summarise(trades):
    if not trades: return None
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    pf = gp/gl if gl > 0 else float("inf")

    # Weekly PnL for consistency check
    week_pnl = {}
    for t in trades:
        w = t["week"]
        week_pnl[w] = week_pnl.get(w, 0) + t["pnl"]
    weeks_positive = sum(1 for v in week_pnl.values() if v > 0)
    weeks_total    = len(week_pnl)

    # First half vs second half (overfitting check)
    mid = len(trades) // 2
    first_pnl  = sum(t["pnl"] for t in trades[:mid])
    second_pnl = sum(t["pnl"] for t in trades[mid:])

    return {
        "n": len(trades), "wr": len(wins)/len(trades)*100, "pf": pf,
        "net": sum(pnls), "net_pct": sum(pnls)/ACCOUNT_EQUITY*100,
        "avg_r": np.mean([t["r_mult"] for t in trades]),
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
        "weeks_pos": weeks_positive, "weeks_total": weeks_total,
        "first_half_pnl": first_pnl, "second_half_pnl": second_pnl,
    }


def main():
    print("Loading all symbols...")
    data = {}
    for sym in ALL_SYMBOLS:
        df = load_and_build(sym)
        if df is not None:
            data[sym] = df

    # Run all combos
    results = []
    for sym in sorted(data.keys()):
        for d in ["long", "short"]:
            trades = run_simple(data[sym], direction=d)
            s = summarise(trades)
            if s is not None:
                results.append({"symbol": sym, "dir": d, "trades": trades, **s})

    # ── CHART 1: Full heatmap — PF by symbol x direction ──
    symbols = sorted(set(r["symbol"] for r in results))
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    # --- Panel 1: Net % return per symbol, long vs short side by side ---
    ax = axes[0, 0]
    x = np.arange(len(symbols))
    width = 0.35
    long_nets  = []
    short_nets = []
    for sym in symbols:
        lr = next((r for r in results if r["symbol"]==sym and r["dir"]=="long"), None)
        sr = next((r for r in results if r["symbol"]==sym and r["dir"]=="short"), None)
        long_nets.append(lr["net_pct"] if lr else 0)
        short_nets.append(sr["net_pct"] if sr else 0)

    bars_l = ax.bar(x - width/2, long_nets,  width, label="LONG",  color=["#4CAF50" if v >= 0 else "#FFCDD2" for v in long_nets], edgecolor="#2E7D32", linewidth=0.8)
    bars_s = ax.bar(x + width/2, short_nets, width, label="SHORT", color=["#F44336" if v >= 0 else "#E0E0E0" for v in short_nets], edgecolor="#C62828", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Net Return (%)")
    ax.set_title("Net Return by Symbol: LONG (green) vs SHORT (red)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()

    # --- Panel 2: Win rate per symbol, long vs short ---
    ax = axes[0, 1]
    long_wr  = []
    short_wr = []
    for sym in symbols:
        lr = next((r for r in results if r["symbol"]==sym and r["dir"]=="long"), None)
        sr = next((r for r in results if r["symbol"]==sym and r["dir"]=="short"), None)
        long_wr.append(lr["wr"] if lr else 0)
        short_wr.append(sr["wr"] if sr else 0)
    ax.bar(x - width/2, long_wr,  width, label="LONG",  color="#81C784", edgecolor="#2E7D32", linewidth=0.8)
    ax.bar(x + width/2, short_wr, width, label="SHORT", color="#EF9A9A", edgecolor="#C62828", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate by Symbol: LONG vs SHORT")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    ax.legend()

    # --- Panel 3: Profit Factor heatmap (bubble chart) ---
    ax = axes[1, 0]
    for i, sym in enumerate(symbols):
        for j, d in enumerate(["long", "short"]):
            r = next((r for r in results if r["symbol"]==sym and r["dir"]==d), None)
            if r is None: continue
            pf = min(r["pf"], 5)  # cap for visual
            n  = r["n"]
            color = "#4CAF50" if r["net"] > 0 else "#F44336"
            alpha = min(0.3 + n/20, 1.0)  # more trades = more opaque
            ax.scatter(i, j, s=pf*200, c=color, alpha=alpha, edgecolors="black", linewidth=0.5)
            label = f"PF={r['pf']:.1f}\nn={r['n']}" if r['pf'] < 10 else f"PF=inf\nn={r['n']}"
            ax.annotate(label, (i, j), textcoords="offset points", xytext=(0, -25),
                       ha="center", fontsize=6, color="black")
    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(symbols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LONG", "SHORT"])
    ax.set_title("Profit Factor (bubble size) x Trade Count (opacity)\nGreen = profitable, Red = losing")

    # --- Panel 4: Overfitting check — first half PnL vs second half PnL ---
    ax = axes[1, 1]
    profitable_results = [r for r in results if r["pf"] > 1.0 and r["n"] >= 5]
    if profitable_results:
        labels = [f"{r['symbol']}\n{r['dir']}" for r in profitable_results]
        first_halves  = [r["first_half_pnl"]  / 1000 for r in profitable_results]
        second_halves = [r["second_half_pnl"] / 1000 for r in profitable_results]
        x2 = np.arange(len(labels))
        ax.bar(x2 - width/2, first_halves,  width, label="First half", color="#90CAF9", edgecolor="#1565C0")
        ax.bar(x2 + width/2, second_halves, width, label="Second half", color="#FFB74D", edgecolor="#E65100")
        ax.set_xticks(x2)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("PnL ($k)")
        ax.set_title("OVERFITTING CHECK: First Half vs Second Half PnL\n(Consistent = both positive, Red flag = one-sided)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()

        # Mark red flags
        for i, r in enumerate(profitable_results):
            if r["first_half_pnl"] < 0 or r["second_half_pnl"] < 0:
                ax.annotate("!", (i, max(first_halves[i], second_halves[i])),
                           fontsize=14, color="red", ha="center", fontweight="bold",
                           xytext=(0, 10), textcoords="offset points")
    else:
        ax.text(0.5, 0.5, "No profitable edges found", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("edge_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: edge_comparison.png")

    # ── CHART 2: Cumulative R-curves for top edges ──
    top_edges = sorted([r for r in results if r["n"] >= 5], key=lambda x: x["pf"], reverse=True)[:12]
    if top_edges:
        fig2, axes2 = plt.subplots(3, 4, figsize=(20, 12), sharey=False)
        axes2_flat = axes2.flatten()
        for idx, r in enumerate(top_edges):
            ax = axes2_flat[idx]
            cum_r = np.cumsum([t["r_mult"] for t in r["trades"]])
            color = "#4CAF50" if cum_r[-1] > 0 else "#F44336"
            ax.plot(cum_r, linewidth=1.5, color=color)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.fill_between(range(len(cum_r)), cum_r, 0, alpha=0.15, color=color)
            pf_str = f"{r['pf']:.1f}" if r['pf'] < 10 else "inf"
            ax.set_title(f"{r['symbol']} {r['dir'].upper()}\nPF={pf_str}  n={r['n']}  WR={r['wr']:.0f}%", fontsize=9)
            ax.set_xlabel("Trade #", fontsize=8)
            ax.set_ylabel("Cumulative R", fontsize=8)

            # Split line at midpoint
            mid = len(cum_r) // 2
            ax.axvline(mid, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)

        for idx in range(len(top_edges), len(axes2_flat)):
            axes2_flat[idx].set_visible(False)
        fig2.suptitle("Cumulative R-Curves (top 12 by PF, orange line = halfway point)", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig("edge_rcurves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: edge_rcurves.png")

    # ── CHART 3: Weekly consistency heatmap ──
    profitable_edges = [r for r in results if r["pf"] > 1.0 and r["n"] >= 5]
    if profitable_edges:
        # Gather all weeks across data
        all_weeks = sorted(set(t["week"] for r in profitable_edges for t in r["trades"]))
        fig3, ax3 = plt.subplots(figsize=(16, max(4, len(profitable_edges)*0.6)))
        matrix = np.zeros((len(profitable_edges), len(all_weeks)))
        for i, r in enumerate(profitable_edges):
            week_pnl = {}
            for t in r["trades"]:
                w = t["week"]
                week_pnl[w] = week_pnl.get(w, 0) + t["r_mult"]
            for j, w in enumerate(all_weeks):
                matrix[i, j] = week_pnl.get(w, float("nan"))

        im = ax3.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=3)
        ax3.set_yticks(range(len(profitable_edges)))
        ax3.set_yticklabels([f"{r['symbol']} {r['dir']}" for r in profitable_edges], fontsize=9)
        ax3.set_xticks(range(len(all_weeks)))
        ax3.set_xticklabels([f"W{w}" for w in all_weeks], fontsize=8)
        ax3.set_title("Weekly R-Return Heatmap (green=profit, red=loss, white=no trades)\nConsistent edges are green across multiple weeks")
        plt.colorbar(im, ax=ax3, label="Weekly R")
        # Annotate NaN cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]):
                    ax3.text(j, i, "-", ha="center", va="center", fontsize=7, color="gray")
                else:
                    ax3.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        plt.savefig("edge_weekly.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: edge_weekly.png")

    # ── Print robustness summary ──
    print(f"\n{'='*90}")
    print(f"  ROBUSTNESS RANKING (profitable edges, sorted by consistency)")
    print(f"{'='*90}")
    print(f"  {'Symbol':<10} {'Dir':<7} {'N':>4} {'WR':>6} {'PF':>6} {'Net%':>8}"
          f" {'WkPos':>7} {'1stHalf':>10} {'2ndHalf':>10} {'Robust?':>9}")
    print(f"  {'-'*86}")

    for r in sorted(profitable_edges, key=lambda x: x["weeks_pos"]/max(x["weeks_total"],1), reverse=True):
        pf_s = f"{r['pf']:.2f}" if r['pf'] < 10 else "inf"
        wk_s = f"{r['weeks_pos']}/{r['weeks_total']}"
        h1   = r["first_half_pnl"]
        h2   = r["second_half_pnl"]
        both_positive = h1 > 0 and h2 > 0
        consistent_weeks = r["weeks_pos"] / max(r["weeks_total"], 1) >= 0.5
        robust = "YES" if both_positive and consistent_weeks and r["n"] >= 8 else \
                 "MAYBE" if both_positive or consistent_weeks else "NO"
        print(
            f"  {r['symbol']:<10} {r['dir']:<7} {r['n']:>4} {r['wr']:>5.0f}% {pf_s:>6} {r['net_pct']:>+7.2f}%"
            f" {wk_s:>7} {h1:>+10,.0f} {h2:>+10,.0f} {robust:>9}"
        )
    print(f"{'='*90}")
    print(f"  Robust = both halves profitable + >=50% winning weeks + >=8 trades")
    print(f"  MAYBE  = one criterion met   |   NO = likely overfitted")


if __name__ == "__main__":
    main()
