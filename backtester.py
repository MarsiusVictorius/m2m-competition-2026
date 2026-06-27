"""
Session Breakout Backtester — Model to Market Competition
Per-symbol strategy configs, pyramiding, and tuned parameters.
"""

import os
import glob
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
DATA_DIR       = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
ACCOUNT_EQUITY = 1_000_000.0
RISK_PER_TRADE = 0.01        # 1% per layer
MAX_POSITIONS  = 3           # portfolio-wide cap
MAX_LEVERAGE   = 25.0
MAX_SINGLE_CONC = 0.85       # only enforced when 2+ positions open
CANDLE_MINUTES = 15

# Available symbols in the dataset
ALL_SYMBOLS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD",
    "GBPUSD", "NZDUSD", "UKOILUSD", "USDCAD", "USDCHF", "USDCNH", "USDHKD",
    "USDJPY", "USOILUSD", "XAGUSD", "XAUCNH", "XAUGCNH", "XAUHKD", "XAUKUSD", "XAUUSD",
]


# ─────────────────────────────────────────────
# PER-SYMBOL STRATEGY CONFIG
# ─────────────────────────────────────────────
@dataclass
class StrategyConfig:
    symbol:         str
    direction:      str   = "both"    # "long", "short", "both"

    # Breakout session: range to break out of
    range_session:  str   = "asia"    # "asia" (00-07) or "london" (07-12)

    # Entry window (UTC hours)
    entry_start:    int   = 7
    entry_end:      int   = 16

    # Hard close hour (UTC)
    session_close:  int   = 21

    # Risk / reward
    sl_atr_mult:    float = 1.5
    tp_atr_mult:    float = 2.0       # relaxed from 3.0 → hits more often

    # Trailing stop
    trail_trigger:  float = 1.0       # ATR profit before trailing activates
    trail_distance: float = 1.0       # ATR behind price for trail
    be_trigger:     float = 1.0       # ATR profit before moving to breakeven

    # Entry quality filters
    confirm_bars:   int   = 1         # consecutive closes needed beyond range (1=classic, 2=confirmed)
    min_break_atr:  float = 0.5       # breakout bar range must be >= N * ATR (momentum filter)
    max_spread_mult:float = 2.0       # skip if spread > N * avg spread

    # Pyramiding (max 3 layers total including first entry)
    pyramid_layers: int   = 3
    pyramid_trigger:float = 1.0       # ATR profit on current layer before adding next


# Proven edge configs from the full scan (PF > 1.5, 5+ trades, fixed sizing).
# Only trade the direction that showed real edge in 30 days of data.
EDGE_CONFIGS: dict[str, StrategyConfig] = {
    "AUDJPY":  StrategyConfig("AUDJPY",  direction="long",  range_session="asia", entry_start=7,  entry_end=16),   # PF inf, WR 88%, +4.6%
    "NZDUSD":  StrategyConfig("NZDUSD",  direction="long",  range_session="asia", entry_start=7,  entry_end=16),   # PF 4.82, WR 90%, +3.8%
    "USDJPY":  StrategyConfig("USDJPY",  direction="short", range_session="asia", entry_start=7,  entry_end=16),   # PF 4.83, WR 83%, +3.8%
    "EURJPY":  StrategyConfig("EURJPY",  direction="short", range_session="asia", entry_start=7,  entry_end=16),   # PF 3.22, WR 73%, +6.7%
    "USDCNH":  StrategyConfig("USDCNH",  direction="long",  range_session="asia", entry_start=0,  entry_end=10),   # PF 2.78, WR 80%, +1.8%
    "AUDUSD":  StrategyConfig("AUDUSD",  direction="short", range_session="asia", entry_start=7,  entry_end=16),   # PF 2.10, WR 73%, +3.3%
    "GBPUSD":  StrategyConfig("GBPUSD",  direction="long",  range_session="asia", entry_start=7,  entry_end=16),   # PF 1.93, WR 58%, +3.7%
    "AUDNZD":  StrategyConfig("AUDNZD",  direction="long",  range_session="asia", entry_start=7,  entry_end=16),   # PF 1.78, WR 73%, +2.3%
    "EURGBP":  StrategyConfig("EURGBP",  direction="short", range_session="asia", entry_start=7,  entry_end=16),   # PF 1.10, WR 59%, +0.6%  (marginal, added for trade count)
    "NZDUSD_s":StrategyConfig("NZDUSD",  direction="short", range_session="asia", entry_start=7,  entry_end=16),   # PF 1.31, WR 58%, +1.6%  (NZD both directions work)
    "XAUKUSD": StrategyConfig("XAUKUSD",  direction="short", range_session="asia", entry_start=7,  entry_end=16),   # PF 1.53, WR 40%, +0.5%
}


# ─────────────────────────────────────────────
# STEP 1: LOAD & BUILD OHLC
# ─────────────────────────────────────────────
def load_tick_files(symbol: str) -> pd.DataFrame:
    pattern = os.path.join(DATA_DIR, f"{symbol}_*.parquet")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files for {symbol} in {DATA_DIR}")
    dfs = [pd.read_parquet(f, columns=["time", "bid", "ask"]) for f in files]
    ticks = pd.concat(dfs, ignore_index=True)
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    return ticks.sort_values("time").reset_index(drop=True)


def build_ohlc(ticks: pd.DataFrame) -> pd.DataFrame:
    ticks = ticks.copy()
    ticks["mid"]    = (ticks["bid"] + ticks["ask"]) / 2
    ticks["spread"] = ticks["ask"] - ticks["bid"]
    ticks = ticks.set_index("time")
    ohlc  = ticks["mid"].resample(f"{CANDLE_MINUTES}min").ohlc()
    ohlc["spread"] = ticks["spread"].resample(f"{CANDLE_MINUTES}min").mean()
    ohlc = ohlc.dropna()
    ohlc.index = ohlc.index.tz_convert("UTC")
    return ohlc


# ─────────────────────────────────────────────
# STEP 2: INDICATORS
# ─────────────────────────────────────────────
def add_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    df = ohlc.copy()

    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["_date"] = df.index.date
    df["_hour"] = df.index.hour

    # Session ranges
    asia   = df[df["_hour"] < 7].groupby("_date").agg(range_high=("high","max"), range_low=("low","min"))
    london = df[(df["_hour"] >= 7) & (df["_hour"] < 12)].groupby("_date").agg(lon_high=("high","max"), lon_low=("low","min"))

    df = df.join(asia,   on="_date")
    df = df.join(london, on="_date")
    df["avg_spread"] = df["spread"].rolling(20).mean()
    df["bar_range"]  = df["high"] - df["low"]   # for momentum filter

    df.drop(columns=["_date", "_hour"], inplace=True)
    return df


# ─────────────────────────────────────────────
# BIAS ANALYSIS
# ─────────────────────────────────────────────
def analyse_bias(symbol: str, df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["sma20", "sma50", "atr14"]).copy()
    d["_hour"] = d.index.hour

    price_return = (d["close"].iloc[-1] - d["close"].iloc[0]) / d["close"].iloc[0] * 100

    daily_close  = d["close"].resample("1D").last().dropna()
    up_day_ratio = (daily_close.diff().dropna() > 0).mean() * 100

    entry_bars = d[(d["_hour"] >= 7) & (d["_hour"] < 16)]
    range_high = df["range_high"].reindex(d.index, method="ffill")
    range_low  = df["range_low"].reindex(d.index, method="ffill")
    breaks_up   = (entry_bars["close"] > range_high.reindex(entry_bars.index)).sum()
    breaks_down = (entry_bars["close"] < range_low.reindex(entry_bars.index)).sum()
    total       = breaks_up + breaks_down
    break_long  = breaks_up / total * 100 if total > 0 else 50

    sma_long = (d["sma20"] > d["sma50"]).mean() * 100

    trend_sig  = np.clip(price_return / 5, -1, 1)
    upday_sig  = (up_day_ratio - 50) / 50
    break_sig  = (break_long - 50) / 50
    sma_sig    = (sma_long - 50) / 50
    composite  = float(np.mean([trend_sig, upday_sig, break_sig, sma_sig]))

    return {
        "symbol":        symbol,
        "return_pct":    price_return,
        "up_day_ratio":  up_day_ratio,
        "break_long_pct":break_long,
        "sma_long_pct":  sma_long,
        "score":         composite,
        "recommended":   "long" if composite > 0.1 else ("short" if composite < -0.1 else "both"),
    }


def print_bias_table(bias_list: list[dict]):
    print("\n" + "=" * 72)
    print("  DIRECTIONAL BIAS ANALYSIS")
    print("=" * 72)
    print(f"  {'Symbol':<12} {'Return':>8} {'UpDays':>8} {'BreakUp':>9} {'SMALong':>9} {'Score':>7}  Bias")
    print("  " + "-" * 68)
    for b in sorted(bias_list, key=lambda x: x["score"]):
        rec = f"SHORT ({b['score']:+.2f})" if b["recommended"] == "short" else \
              f"LONG  ({b['score']:+.2f})" if b["recommended"] == "long"  else \
              f"BOTH  ({b['score']:+.2f})"
        print(
            f"  {b['symbol']:<12}"
            f" {b['return_pct']:>+7.2f}%"
            f" {b['up_day_ratio']:>7.1f}%"
            f" {b['break_long_pct']:>8.1f}%"
            f" {b['sma_long_pct']:>8.1f}%"
            f"   {rec}"
        )
    print("=" * 72)


# ─────────────────────────────────────────────
# TRADE DATACLASS
# ─────────────────────────────────────────────
@dataclass
class Trade:
    symbol:       str
    direction:    str
    layer:        int             # 1, 2, or 3 (pyramid layer)
    entry_time:   pd.Timestamp
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    atr_at_entry: float
    size:         float
    exit_time:    Optional[pd.Timestamp] = None
    exit_price:   Optional[float]        = None
    pnl:          float = 0.0
    mae:          float = 0.0
    mfe:          float = 0.0
    exit_reason:  str   = ""
    be_moved:     bool  = False
    trailing:     bool  = False
    group_id:     str   = ""      # ties pyramid layers together

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def unrealised_pnl(self, price: float) -> float:
        if self.direction == "long":
            return (price - self.entry_price) * self.size
        return (self.entry_price - price) * self.size

    def profit_in_atr(self, price: float) -> float:
        raw = (price - self.entry_price) if self.direction == "long" \
              else (self.entry_price - price)
        return raw / self.atr_at_entry if self.atr_at_entry > 0 else 0

    def update_mae_mfe(self, price: float):
        u = self.unrealised_pnl(price)
        if u < self.mae: self.mae = u
        if u > self.mfe: self.mfe = u


# ─────────────────────────────────────────────
# POSITION SIZING & RISK CHECKS
# ─────────────────────────────────────────────
def position_size(stop_distance: float, equity: float) -> float:
    if stop_distance <= 0:
        return 0.0
    return (equity * RISK_PER_TRADE) / stop_distance


def check_leverage(open_trades: list[Trade], new_size: float, new_price: float, equity: float) -> bool:
    total = sum(abs(t.size * t.entry_price) for t in open_trades) + abs(new_size * new_price)
    return (total / equity) <= MAX_LEVERAGE


def check_concentration(open_trades: list[Trade], symbol: str, new_size: float, new_price: float, equity: float) -> bool:
    if len(open_trades) < 2:
        return True
    sym   = sum(abs(t.size * t.entry_price) for t in open_trades if t.symbol == symbol) + abs(new_size * new_price)
    total = sum(abs(t.size * t.entry_price) for t in open_trades) + abs(new_size * new_price)
    return (sym / total) <= MAX_SINGLE_CONC


# ─────────────────────────────────────────────
# UNIFIED PORTFOLIO ENGINE
# ─────────────────────────────────────────────
def run_portfolio(dfs: dict[str, pd.DataFrame],
                  configs: dict[str, StrategyConfig]) -> tuple[list[Trade], pd.Series]:
    """
    Walk the unified timeline once. At each bar, process all symbols using
    their per-symbol config. Equity, open positions, and leverage limits
    are shared across the whole portfolio in real time.
    """
    all_times = sorted(set().union(*[set(df.index) for df in dfs.values()]))
    equity_curve = pd.Series(index=all_times, dtype=float)

    equity       = ACCOUNT_EQUITY
    open_trades: list[Trade]  = []
    closed_trades: list[Trade] = []

    # Per-symbol state
    confirm_long:   dict[str, int] = {s: 0 for s in dfs}
    confirm_short:  dict[str, int] = {s: 0 for s in dfs}
    session_entries: dict[tuple, bool] = {}
    pyramid_groups:  dict[str, list[Trade]] = {}
    group_counter = 0

    for ts in all_times:
        hour     = ts.hour
        date_str = ts.date()

        for symbol, df in dfs.items():
            if ts not in df.index:
                continue
            bar = df.loc[ts]
            cfg = configs.get(symbol, StrategyConfig(symbol))

            if pd.isna(bar["atr14"]) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]):
                continue

            close      = bar["close"]
            high       = bar["high"]
            low        = bar["low"]
            atr        = bar["atr14"]
            sma20      = bar["sma20"]
            sma50      = bar["sma50"]
            spread     = bar["spread"]
            avg_spread = bar["avg_spread"] if "avg_spread" in bar.index and not pd.isna(bar["avg_spread"]) else spread
            bar_rng    = bar["bar_range"]

            if cfg.range_session == "london":
                rng_high = bar["lon_high"]   if "lon_high"   in bar.index else float("nan")
                rng_low  = bar["lon_low"]    if "lon_low"    in bar.index else float("nan")
            else:
                rng_high = bar["range_high"] if "range_high" in bar.index else float("nan")
                rng_low  = bar["range_low"]  if "range_low"  in bar.index else float("nan")

            sym_open = [t for t in open_trades if t.symbol == symbol]

            # ── UPDATE open trades ──
            for trade in sym_open:
                trade.update_mae_mfe(close)

                if not trade.be_moved and trade.profit_in_atr(close) >= cfg.be_trigger:
                    trade.stop_loss = trade.entry_price
                    trade.be_moved  = True

                if not trade.trailing and trade.profit_in_atr(close) >= cfg.trail_trigger:
                    trade.trailing = True

                if trade.trailing:
                    if trade.direction == "long":
                        trade.stop_loss = max(trade.stop_loss, close - cfg.trail_distance * atr)
                    else:
                        trade.stop_loss = min(trade.stop_loss, close + cfg.trail_distance * atr)

                hit_sl = (trade.direction == "long"  and low  <= trade.stop_loss) or \
                         (trade.direction == "short" and high >= trade.stop_loss)
                hit_tp = (trade.direction == "long"  and high >= trade.take_profit) or \
                         (trade.direction == "short" and low  <= trade.take_profit)

                if hit_tp:
                    trade.exit_price  = trade.take_profit
                    trade.exit_time   = ts
                    trade.exit_reason = "TP"
                elif hit_sl:
                    trade.exit_price  = trade.stop_loss
                    trade.exit_time   = ts
                    trade.exit_reason = "SL"

            # ── Hard close at session end ──
            if hour == cfg.session_close:
                for trade in [t for t in sym_open if t.is_open]:
                    trade.exit_price  = close
                    trade.exit_time   = ts
                    trade.exit_reason = "TIME"

            # ── Settle closed trades ──
            for trade in [t for t in open_trades if t.symbol == symbol and not t.is_open]:
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.size \
                            if trade.direction == "long" \
                            else (trade.entry_price - trade.exit_price) * trade.size
                equity += trade.pnl
                closed_trades.append(trade)
                gid = trade.group_id
                if gid in pyramid_groups:
                    pyramid_groups[gid] = [t for t in pyramid_groups[gid] if t.is_open]
                    if not pyramid_groups[gid]:
                        del pyramid_groups[gid]
            open_trades = [t for t in open_trades if t.is_open]

            # ── PYRAMID: add to winning positions ──
            for gid in [g for g in pyramid_groups if g.startswith(symbol)]:
                layers = pyramid_groups[gid]
                if not layers or len(layers) >= cfg.pyramid_layers:
                    continue
                last_layer = layers[-1]
                if last_layer.profit_in_atr(close) >= cfg.pyramid_trigger:
                    for t in layers:
                        t.stop_loss = t.entry_price
                        t.be_moved  = True
                    direction = layers[0].direction
                    sl = close - cfg.sl_atr_mult * atr if direction == "long" else close + cfg.sl_atr_mult * atr
                    tp = close + cfg.tp_atr_mult * atr if direction == "long" else close - cfg.tp_atr_mult * atr
                    sz = position_size(abs(close - sl), equity)
                    if sz > 0 \
                            and len(open_trades) < MAX_POSITIONS \
                            and check_leverage(open_trades, sz, close, equity) \
                            and check_concentration(open_trades, symbol, sz, close, equity):
                        group_counter += 1
                        new_trade = Trade(
                            symbol=symbol, direction=direction, layer=len(layers) + 1,
                            entry_time=ts, entry_price=close,
                            stop_loss=sl, take_profit=tp,
                            atr_at_entry=atr, size=sz, group_id=gid,
                        )
                        open_trades.append(new_trade)
                        layers.append(new_trade)

            # ── ENTRY LOGIC ──
            if not (cfg.entry_start <= hour < cfg.entry_end):
                confirm_long[symbol]  = 0
                confirm_short[symbol] = 0
                continue

            if pd.isna(rng_high) or pd.isna(rng_low):
                continue
            if spread > cfg.max_spread_mult * avg_spread:
                continue

            confirm_long[symbol]  = (confirm_long[symbol]  + 1) if close > rng_high else 0
            confirm_short[symbol] = (confirm_short[symbol] + 1) if close < rng_low  else 0

            long_key  = (symbol, date_str, "long")
            short_key = (symbol, date_str, "short")

            # ── LONG entry ──
            if cfg.direction in ("long", "both") \
                    and confirm_long[symbol] >= cfg.confirm_bars \
                    and close > sma20 and sma20 > sma50 \
                    and bar_rng >= cfg.min_break_atr * atr \
                    and long_key not in session_entries \
                    and len(open_trades) < MAX_POSITIONS:
                sl   = close - cfg.sl_atr_mult * atr
                tp   = close + cfg.tp_atr_mult * atr
                size = position_size(close - sl, equity)
                if size > 0 \
                        and check_leverage(open_trades, size, close, equity):
                    group_counter += 1
                    gid = f"{symbol}_L{group_counter}"
                    trade = Trade(
                        symbol=symbol, direction="long", layer=1,
                        entry_time=ts, entry_price=close,
                        stop_loss=sl, take_profit=tp,
                        atr_at_entry=atr, size=size, group_id=gid,
                    )
                    open_trades.append(trade)
                    pyramid_groups[gid] = [trade]
                    session_entries[long_key] = True

            # ── SHORT entry ──
            if cfg.direction in ("short", "both") \
                    and confirm_short[symbol] >= cfg.confirm_bars \
                    and close < sma20 and sma20 < sma50 \
                    and bar_rng >= cfg.min_break_atr * atr \
                    and short_key not in session_entries \
                    and len(open_trades) < MAX_POSITIONS:
                sl   = close + cfg.sl_atr_mult * atr
                tp   = close - cfg.tp_atr_mult * atr
                size = position_size(sl - close, equity)
                if size > 0 \
                        and check_leverage(open_trades, size, close, equity):
                    group_counter += 1
                    gid = f"{symbol}_S{group_counter}"
                    trade = Trade(
                        symbol=symbol, direction="short", layer=1,
                        entry_time=ts, entry_price=close,
                        stop_loss=sl, take_profit=tp,
                        atr_at_entry=atr, size=size, group_id=gid,
                    )
                    open_trades.append(trade)
                    pyramid_groups[gid] = [trade]
                    session_entries[short_key] = True

        # ── Equity snapshot (realised + unrealised) ──
        unrealised = 0.0
        for t in open_trades:
            if t.symbol in dfs and ts in dfs[t.symbol].index:
                unrealised += t.unrealised_pnl(dfs[t.symbol].loc[ts, "close"])
        equity_curve[ts] = equity + unrealised

    # Force-close anything open at end of data
    for trade in list(open_trades):
        last_price = dfs[trade.symbol]["close"].iloc[-1]
        trade.exit_price  = last_price
        trade.exit_time   = all_times[-1]
        trade.exit_reason = "EOD"
        trade.pnl = (trade.exit_price - trade.entry_price) * trade.size \
                    if trade.direction == "long" \
                    else (trade.entry_price - trade.exit_price) * trade.size
        equity += trade.pnl
        closed_trades.append(trade)

    return closed_trades, equity_curve


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────
def compute_metrics(trades: list[Trade], equity_curve: pd.Series, label: str = "") -> dict:
    if not trades:
        return {}
    pnls   = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp     = sum(wins)        if wins   else 0.0
    gl     = abs(sum(losses)) if losses else 0.0
    pf     = gp / gl if gl > 0 else float("inf")

    eq  = equity_curve.dropna()
    dd  = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - ACCOUNT_EQUITY) / ACCOUNT_EQUITY * 100
    r   = eq.pct_change().dropna()
    sh  = float(r.mean() / r.std()) if r.std() > 0 else 0.0

    return {
        "label":           label,
        "total_trades":    len(trades),
        "win_rate":        len(wins) / len(trades) * 100,
        "avg_win":         float(np.mean(wins))   if wins   else 0.0,
        "avg_loss":        float(np.mean(losses)) if losses else 0.0,
        "profit_factor":   pf,
        "net_return_pct":  ret,
        "max_drawdown_pct":dd,
        "sharpe_15min":    sh,
        "gross_profit":    gp,
        "gross_loss":      gl,
    }


def print_summary(m: dict):
    print("\n" + "=" * 52)
    print(f"  PERFORMANCE — {m.get('label','')}")
    print("=" * 52)
    rows = [
        ("Total trades",    f"{m['total_trades']}"),
        ("Win rate",        f"{m['win_rate']:.1f}%"),
        ("Avg win",         f"${m['avg_win']:,.2f}"),
        ("Avg loss",        f"${m['avg_loss']:,.2f}"),
        ("Profit factor",   f"{m['profit_factor']:.2f}"),
        ("Net return",      f"{m['net_return_pct']:+.2f}%"),
        ("Max drawdown",    f"{m['max_drawdown_pct']:.2f}%"),
        ("Sharpe (15-min)", f"{m['sharpe_15min']:.4f}"),
        ("Gross profit",    f"${m['gross_profit']:,.2f}"),
        ("Gross loss",      f"${m['gross_loss']:,.2f}"),
    ]
    for lbl, val in rows:
        print(f"  {lbl:<22} {val}")
    print("=" * 52)


def print_per_symbol(all_trades: list[Trade], equity_curve: pd.Series):
    symbols = sorted(set(t.symbol for t in all_trades))
    print("\n" + "=" * 80)
    print("  PER-SYMBOL BREAKDOWN")
    print("=" * 80)
    print(f"  {'Symbol':<12} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Return':>9} {'MaxDD':>8} {'AvgWin':>10} {'AvgLoss':>10}")
    print("  " + "-" * 76)
    for sym in symbols:
        sym_trades = [t for t in all_trades if t.symbol == sym]
        if not sym_trades:
            continue
        pnls   = [t.pnl for t in sym_trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gp     = sum(wins)        if wins   else 0.0
        gl     = abs(sum(losses)) if losses else 0.0
        pf     = gp / gl if gl > 0 else float("inf")
        wr     = len(wins) / len(pnls) * 100
        net    = sum(pnls)
        net_p  = net / ACCOUNT_EQUITY * 100
        aw     = float(np.mean(wins))   if wins   else 0.0
        al     = float(np.mean(losses)) if losses else 0.0
        pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
        print(
            f"  {sym:<12}"
            f" {len(pnls):>7}"
            f" {wr:>7.1f}%"
            f" {pf_str:>6}"
            f" {net_p:>+8.2f}%"
            f" {'N/A':>8}"
            f" {aw:>10,.0f}"
            f" {al:>10,.0f}"
        )
    print("=" * 80)


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
def save_trade_log(trades: list[Trade], path: str = "trade_log.csv"):
    rows = [{
        "entry_time":  t.entry_time,
        "exit_time":   t.exit_time,
        "symbol":      t.symbol,
        "direction":   t.direction,
        "layer":       t.layer,
        "group_id":    t.group_id,
        "entry_price": t.entry_price,
        "exit_price":  t.exit_price,
        "size":        t.size,
        "pnl":         t.pnl,
        "mae":         t.mae,
        "mfe":         t.mfe,
        "exit_reason": t.exit_reason,
    } for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Trade log saved: {path}")


def plot_results(trades: list[Trade], equity_curve: pd.Series, filename: str = "backtest_results.png"):
    symbols = sorted(set(t.symbol for t in trades))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(symbols)))
    sym_color = {s: c for s, c in zip(symbols, colors)}

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    ax_eq  = fig.add_subplot(gs[0, :])
    ax_mae = fig.add_subplot(gs[1, 0])
    ax_mfe = fig.add_subplot(gs[1, 1])
    ax_sym = fig.add_subplot(gs[1, 2])

    # Equity curve
    eq = equity_curve.dropna()
    ax_eq.plot(eq.index, eq.values, linewidth=1.2, color="#2196F3", label="Portfolio")
    ax_eq.axhline(ACCOUNT_EQUITY, color="gray", linestyle="--", linewidth=0.8)
    ax_eq.set_title("Portfolio Equity Curve")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    pnls   = np.array([t.pnl for t in trades])
    maes   = np.array([t.mae for t in trades])
    mfes   = np.array([t.mfe for t in trades])
    tcolors = [sym_color[t.symbol] for t in trades]

    ax_mae.scatter(maes, pnls, c=tcolors, alpha=0.7, edgecolors="none", s=40)
    ax_mae.axhline(0, color="gray", linewidth=0.8)
    ax_mae.axvline(0, color="gray", linewidth=0.8)
    ax_mae.set_title("MAE vs Final PnL")
    ax_mae.set_xlabel("MAE ($)")
    ax_mae.set_ylabel("PnL ($)")

    ax_mfe.scatter(mfes, pnls, c=tcolors, alpha=0.7, edgecolors="none", s=40)
    ax_mfe.axhline(0, color="gray", linewidth=0.8)
    ax_mfe.axvline(0, color="gray", linewidth=0.8)
    ax_mfe.set_title("MFE vs Final PnL")
    ax_mfe.set_xlabel("MFE ($)")
    ax_mfe.set_ylabel("PnL ($)")

    # Per-symbol PnL bar chart
    sym_pnl = {s: sum(t.pnl for t in trades if t.symbol == s) for s in symbols}
    bar_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in sym_pnl.values()]
    ax_sym.bar(sym_pnl.keys(), sym_pnl.values(), color=bar_colors)
    ax_sym.axhline(0, color="gray", linewidth=0.8)
    ax_sym.set_title("PnL by Symbol")
    ax_sym.set_ylabel("Total PnL ($)")
    ax_sym.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Charts saved: {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Load data for all edge symbols ──
    needed_symbols = sorted(set(cfg.symbol for cfg in EDGE_CONFIGS.values()))
    print(f"\nLoading data for: {needed_symbols}")
    dfs = {}
    for symbol in needed_symbols:
        try:
            ticks = load_tick_files(symbol)
        except FileNotFoundError:
            print(f"  [{symbol}] no data -- skipping")
            continue
        ohlc = build_ohlc(ticks)
        ohlc = add_indicators(ohlc)
        dfs[symbol] = ohlc
        print(f"  [{symbol}] {len(ohlc)} candles")

    # ── Bias analysis (informational) ──
    print()
    bias_list = [analyse_bias(sym, df) for sym, df in dfs.items()]
    print_bias_table(bias_list)

    # ── Use the proven edge configs directly (no bias override) ──
    # Filter out configs whose symbol has no data, and deduplicate by symbol
    # (NZDUSD appears twice — "both" directions — use the main long config,
    #  the engine will only trade the direction specified in each config)
    configs: dict[str, StrategyConfig] = {}
    for key, cfg in EDGE_CONFIGS.items():
        if cfg.symbol in dfs and cfg.symbol not in configs:
            configs[cfg.symbol] = cfg

    print("\nPer-symbol strategy config (edge-only):")
    print(f"  {'Symbol':<12} {'Direction':<10} {'Session':<10} {'EntryWin':<12} {'TP mult':<9} {'SL mult':<9} {'Pyramid'}")
    print("  " + "-" * 68)
    for sym, cfg in configs.items():
        print(
            f"  {sym:<12} {cfg.direction:<10} {cfg.range_session:<10}"
            f" {cfg.entry_start:02d}:00-{cfg.entry_end:02d}:00   "
            f" {cfg.tp_atr_mult:<9.1f} {cfg.sl_atr_mult:<9.1f} {cfg.pyramid_layers}"
        )

    # ── Run portfolio ──
    print("\nRunning portfolio strategy...")
    trades, equity_curve = run_portfolio(dfs, configs)
    print(f"  {len(trades)} trades completed across {len(dfs)} symbols")

    if not trades:
        print("No trades generated.")
        return

    metrics = compute_metrics(trades, equity_curve, label="EDGE PORTFOLIO")
    print_summary(metrics)
    print_per_symbol(trades, equity_curve)

    save_trade_log(trades)
    plot_results(trades, equity_curve)


if __name__ == "__main__":
    main()
