"""Quick edge scan on Symphonix-available symbols we haven't tested yet."""
import os, glob, warnings
from dataclasses import dataclass
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
ACCOUNT_EQUITY = 1_000_000.0; FIXED_RISK = 10_000.0; MAX_POSITIONS = 6
SLIPPAGE_SPREAD_MULT = 0.5
SL_ATR = 1.5; TP_ATR = 1.6; BE_ATR = 1.0; TRAIL_ON = 1.0; TRAIL_D = 1.0

def load(symbol, cm=15):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{symbol}_*.parquet")))
    if not files: return None
    ticks = pd.concat([pd.read_parquet(f, columns=["time","bid","ask"]) for f in files], ignore_index=True)
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    ticks = ticks.sort_values("time")
    ticks["mid"] = (ticks["bid"]+ticks["ask"])/2
    ticks["spread"] = ticks["ask"]-ticks["bid"]
    ticks = ticks.set_index("time")
    o = ticks["mid"].resample(f"{cm}min").ohlc()
    o["spread"] = ticks["spread"].resample(f"{cm}min").mean()
    o = o.dropna(); o.index = o.index.tz_convert("UTC")
    o["sma20"]=o["close"].rolling(20).mean(); o["sma50"]=o["close"].rolling(50).mean()
    p=o["close"].shift(1)
    tr=pd.concat([o["high"]-o["low"],(o["high"]-p).abs(),(o["low"]-p).abs()],axis=1).max(axis=1)
    o["atr14"]=tr.rolling(14).mean(); o["atr5"]=tr.rolling(5).mean(); o["atr40"]=tr.rolling(40).mean()
    o["avg_spread"]=o["spread"].rolling(20).mean()
    o["ema20"]=o["close"].ewm(span=20,adjust=False).mean()
    o["_date"]=o.index.date; o["_hour"]=o.index.hour
    daily=o.groupby("_date").agg(day_open=("open","first"),day_close=("close","last"))
    daily["prev_close"]=daily["day_close"].shift(1)
    o=o.join(daily[["day_open","prev_close"]],on="_date")
    asia=o[o["_hour"]<7].groupby("_date").agg(asia_high=("high","max"),asia_low=("low","min"))
    o=o.join(asia,on="_date")
    o["highest_h10"]=o["high"].rolling(10).max(); o["lowest_l10"]=o["low"].rolling(10).min()
    o.drop(columns=["_date","_hour"],inplace=True)
    return o

@dataclass
class Trade:
    symbol:str;direction:str;entry_time:object;entry_price:float
    stop_loss:float;take_profit:float;atr:float;size:float;config_label:str
    exit_time:object=None;exit_price:float=None;pnl:float=0.0;exit_reason:str=""
    be_moved:bool=False;trailing:bool=False
    @property
    def is_open(self):return self.exit_time is None
    def profit_atr(self,p):
        r=(p-self.entry_price) if self.direction=="long" else(self.entry_price-p)
        return r/self.atr if self.atr>0 else 0

# Test configs: all POI types × both directions × all T-segments × all filters
POI_TYPES = ["asia", "prev_close"]
FRACTS = [0.0, 0.3, 0.5, 1.0]
FILTERS = ["none", "vol_contracting", "vol_expanding", "pullback_atr"]
T_SEGMENTS = [0, 1, 2, 3]

def get_poi(bar, poi_type):
    if poi_type == "asia":
        return bar.get("asia_high", np.nan), bar.get("asia_low", np.nan)
    elif poi_type == "prev_close":
        pc = bar.get("prev_close", np.nan)
        return pc, pc
    return np.nan, np.nan

def check_filter(bar, f, d):
    if f == "none": return True
    elif f == "vol_expanding": return bar["atr5"] > bar["atr40"]
    elif f == "vol_contracting": return bar["atr5"] <= bar["atr40"]
    elif f == "pullback_atr":
        if d == "long": return bar["atr14"] > bar.get("prev_close", bar["close"]) - bar["lowest_l10"]
        return bar["atr14"] > bar["highest_h10"] - bar.get("prev_close", bar["close"])
    return True

def run_single(df, symbol, direction, poi_type, fract, filter1, t_seg):
    label = f"{symbol} {direction} {poi_type}+{fract}+{filter1}+T{t_seg}"
    eq = ACCOUNT_EQUITY; ot = []; cl = []; entries_done = {}

    for ts, bar in df.iterrows():
        if pd.isna(bar["atr14"]) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]): continue
        hi = ts.hour; h = hi + ts.minute/60.0
        c, hh, ll = bar["close"], bar["high"], bar["low"]
        atr = bar["atr14"]; sp = bar["spread"]
        avg_sp = bar["avg_spread"] if not pd.isna(bar.get("avg_spread", np.nan)) else sp
        date = ts.date()

        es, ee = 7, 16; tl = (ee-es)/3
        if t_seg == 1: ss, se = es, es+tl
        elif t_seg == 2: ss, se = es+tl, es+2*tl
        elif t_seg == 3: ss, se = es+2*tl, ee
        else: ss, se = es, ee

        # Manage open trades
        for t in list(ot):
            if not t.be_moved and t.profit_atr(c) >= BE_ATR:
                t.stop_loss = t.entry_price; t.be_moved = True
            if not t.trailing and t.profit_atr(c) >= TRAIL_ON:
                t.trailing = True
            if t.trailing:
                if t.direction == "long": t.stop_loss = max(t.stop_loss, c - TRAIL_D*atr)
                else: t.stop_loss = min(t.stop_loss, c + TRAIL_D*atr)

            hsl = (t.direction=="long" and ll<=t.stop_loss) or (t.direction=="short" and hh>=t.stop_loss)
            htp = (t.direction=="long" and hh>=t.take_profit) or (t.direction=="short" and ll<=t.take_profit)
            slip = SLIPPAGE_SPREAD_MULT * sp

            if htp:
                t.exit_price = (t.take_profit-slip if t.direction=="long" else t.take_profit+slip)
                t.exit_time = ts; t.exit_reason = "TP"
            elif hsl:
                t.exit_price = (t.stop_loss-slip if t.direction=="long" else t.stop_loss+slip)
                t.exit_time = ts; t.exit_reason = "SL"

        if hi == 21:
            for t in [x for x in ot if x.is_open]:
                slip = SLIPPAGE_SPREAD_MULT * sp
                t.exit_price = (c-slip if t.direction=="long" else c+slip)
                t.exit_time = ts; t.exit_reason = "TIME"

        for t in [x for x in ot if not x.is_open]:
            t.pnl = (t.exit_price-t.entry_price)*t.size if t.direction=="long" else (t.entry_price-t.exit_price)*t.size
            eq += t.pnl; cl.append(t)
        ot = [t for t in ot if t.is_open]

        # Entry logic
        if not (ss <= h < se): continue
        if pd.isna(avg_sp) or sp > 2.0*avg_sp: continue

        pl, ps = get_poi(bar, poi_type)
        if pd.isna(pl) or pd.isna(ps): continue
        bl = pl + fract*atr if fract > 0 else pl
        bs = ps - fract*atr if fract > 0 else ps

        ek = (label, date)
        if ek in entries_done: continue

        if direction == "long" and c > bl and c > bar["sma20"] and bar["sma20"] > bar["sma50"]:
            if check_filter(bar, filter1, "long") and len(ot) < MAX_POSITIONS:
                slip = SLIPPAGE_SPREAD_MULT*sp; ep = c+slip
                sl = ep - SL_ATR*atr; tp = ep + TP_ATR*atr
                sz = FIXED_RISK/(SL_ATR*atr)
                if sz <= 0: continue
                t = Trade(symbol,"long",ts,ep,sl,tp,atr,sz,label)
                ot.append(t); entries_done[ek] = True

        elif direction == "short" and c < bs and c < bar["sma20"] and bar["sma20"] < bar["sma50"]:
            if check_filter(bar, filter1, "short") and len(ot) < MAX_POSITIONS:
                slip = SLIPPAGE_SPREAD_MULT*sp; ep = c-slip
                sl = ep + SL_ATR*atr; tp = ep - TP_ATR*atr
                sz = FIXED_RISK/(SL_ATR*atr)
                if sz <= 0: continue
                t = Trade(symbol,"short",ts,ep,sl,tp,atr,sz,label)
                ot.append(t); entries_done[ek] = True

    # Close remaining
    for t in list(ot):
        last_bar = df.iloc[-1]
        t.exit_price = last_bar["close"]; t.exit_time = df.index[-1]; t.exit_reason = "EOD"
        t.pnl = (t.exit_price-t.entry_price)*t.size if t.direction=="long" else (t.entry_price-t.exit_price)*t.size
        eq += t.pnl; cl.append(t)

    pnls = [t.pnl for t in cl]
    if not pnls: return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0.001
    pf = gp/gl if gl > 0 else float("inf")
    wr = len(wins)/len(pnls)*100
    net = sum(pnls)/ACCOUNT_EQUITY*100
    return {"label": label, "trades": len(pnls), "wr": wr, "pf": pf, "net_pct": net}

# Scan these symbols
SCAN_SYMBOLS = ["EURUSD", "USDCAD", "USDCHF", "EURCHF", "XAGUSD"]

print("="*80)
print("EDGE SCAN: New symbols available on Symphonix broker")
print("="*80)

results = []
for sym in SCAN_SYMBOLS:
    print(f"\nLoading {sym}...")
    df = load(sym)
    if df is None:
        print(f"  No data for {sym}")
        continue
    print(f"  {len(df)} bars loaded, {df.index[0].date()} to {df.index[-1].date()}")

    for direction in ["long", "short"]:
        for poi in POI_TYPES:
            for fract in FRACTS:
                for filt in FILTERS:
                    for tseg in T_SEGMENTS:
                        r = run_single(df, sym, direction, poi, fract, filt, tseg)
                        if r and r["trades"] >= 3 and r["pf"] > 1.5 and r["wr"] >= 60:
                            results.append(r)

# Sort by profit factor
results.sort(key=lambda x: x["pf"], reverse=True)

print(f"\n{'='*80}")
print(f"RESULTS: {len(results)} configs with trades>=3, PF>1.5, WR>=60%")
print(f"{'='*80}")
print(f"{'Config':<55} {'Trades':>6} {'WR%':>6} {'PF':>7} {'Net%':>7}")
print("-"*85)
for r in results[:30]:
    pf_s = f"{r['pf']:.2f}" if r['pf'] < 100 else "inf"
    print(f"{r['label']:<55} {r['trades']:>6} {r['wr']:>5.1f}% {pf_s:>7} {r['net_pct']:>+6.1f}%")
