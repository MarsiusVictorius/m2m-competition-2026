"""Quick timeframe comparison: M15 vs M30 vs H1"""
import os, glob, warnings
from dataclasses import dataclass
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = r"C:\Users\marti\Documents\Claude Apps\App 101\Data for backtests"
ACCOUNT_EQUITY = 1_000_000.0; FIXED_RISK = 10_000.0; MAX_POSITIONS = 6
TARGET_MAX_LEVERAGE = 20.0; SLIPPAGE_SPREAD_MULT = 0.5
SL_ATR = 1.5; TP_ATR = 1.6; BE_ATR = 1.0; TRAIL_ON = 1.0; TRAIL_D = 1.0
PYRAMID_LAYERS = 3; PYRAMID_TRIGGER = 1.0

@dataclass
class C:
    symbol:str;direction:str;poi_type:str="asia";fract:float=0.0;filter1:str="none";t_segment:int=0;label:str=""

CONFIGS = [
    C("AUDJPY","long","asia",0.0,"none",0,"AUDJPY long asia"),
    C("AUDNZD","long","asia",0.0,"vol_contracting",3,"AUDNZD long asia+volC+T3"),
    C("AUDUSD","short","asia",0.0,"vol_contracting",2,"AUDUSD short asia+volC+T2"),
    C("EURGBP","short","asia",0.3,"none",1,"EURGBP short asia+0.3+T1"),
    C("EURJPY","short","prev_close",1.0,"pullback_atr",0,"EURJPY short prevC+pull"),
    C("GBPUSD","long","asia",0.0,"none",1,"GBPUSD long asia+T1"),
    C("NZDUSD","long","asia",0.0,"none",1,"NZDUSD long asia+T1"),
    C("USDCNH","long","asia",0.0,"vol_contracting",0,"USDCNH long asia+volC"),
    C("USDJPY","short","prev_close",0.5,"vol_expanding",1,"USDJPY short prevC+volE+T1"),
    C("USDJPY","long","prev_close",0.5,"vol_expanding",2,"USDJPY long prevC+volE+T2"),
    C("XAUKUSD","short","prev_close",0.5,"none",0,"XAUKUSD short prevC+0.5"),
]

def load(symbol, cm):
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
    up=o["high"]-o["high"].shift(1); dn=o["low"].shift(1)-o["low"]
    pdm=np.where((up>dn)&(up>0),up,0.0); mdm=np.where((dn>up)&(dn>0),dn,0.0)
    a14s=tr.ewm(span=14,adjust=False).mean()
    o["dmi_plus"]=pd.Series(pdm,index=o.index).ewm(span=14,adjust=False).mean()/a14s*100
    o["dmi_minus"]=pd.Series(mdm,index=o.index).ewm(span=14,adjust=False).mean()/a14s*100
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

def get_poi(bar,t):
    if t=="asia": return bar.get("asia_high",np.nan),bar.get("asia_low",np.nan)
    elif t=="prev_close": pc=bar.get("prev_close",np.nan);return pc,pc
    elif t=="cum_ma": cm=bar.get("cum_ma",np.nan);return cm,cm
    elif t=="ema20": e=bar.get("ema20",np.nan);return e,e
    return np.nan,np.nan

def chk(bar,f,d):
    if f=="none":return True
    elif f=="vol_expanding":return bar["atr5"]>bar["atr40"]
    elif f=="vol_contracting":return bar["atr5"]<=bar["atr40"]
    elif f=="pullback_atr":
        if d=="long":return bar["atr14"]>bar.get("prev_close",bar["close"])-bar["lowest_l10"]
        return bar["atr14"]>bar["highest_h10"]-bar.get("prev_close",bar["close"])
    return True

@dataclass
class T:
    symbol:str;direction:str;layer:int;group_id:str;entry_time:object;entry_price:float
    stop_loss:float;take_profit:float;atr:float;size:float;config_label:str
    exit_time:object=None;exit_price:float=None;pnl:float=0.0;exit_reason:str=""
    be_moved:bool=False;trailing:bool=False
    @property
    def is_open(self):return self.exit_time is None
    def profit_atr(self,p):r=(p-self.entry_price) if self.direction=="long" else(self.entry_price-p);return r/self.atr if self.atr>0 else 0

def run(dfs):
    times=sorted(set().union(*[set(df.index) for df in dfs.values()]))
    eq=ACCOUNT_EQUITY;ot=[];cl=[];ent={};pyr={};gc=0
    for ts in times:
        hi=ts.hour;h=hi+ts.minute/60.0;date=ts.date()
        for cfg in CONFIGS:
            sym=cfg.symbol
            if sym not in dfs or ts not in dfs[sym].index:continue
            bar=dfs[sym].loc[ts]
            if pd.isna(bar["atr14"]) or pd.isna(bar["sma20"]) or pd.isna(bar["sma50"]):continue
            c,hh,ll=bar["close"],bar["high"],bar["low"];atr=bar["atr14"];sp=bar["spread"]
            avg_sp=bar["avg_spread"] if not pd.isna(bar.get("avg_spread",np.nan)) else sp
            es,ee=7,16;tl=(ee-es)/3
            if cfg.t_segment==1:ss,se=es,es+tl
            elif cfg.t_segment==2:ss,se=es+tl,es+2*tl
            elif cfg.t_segment==3:ss,se=es+2*tl,ee
            else:ss,se=es,ee
            so=[t for t in ot if t.config_label==cfg.label]
            for t in so:
                if not t.be_moved and t.profit_atr(c)>=BE_ATR:t.stop_loss=t.entry_price;t.be_moved=True
                if not t.trailing and t.profit_atr(c)>=TRAIL_ON:t.trailing=True
                if t.trailing:
                    if t.direction=="long":t.stop_loss=max(t.stop_loss,c-TRAIL_D*atr)
                    else:t.stop_loss=min(t.stop_loss,c+TRAIL_D*atr)
                hsl=(t.direction=="long" and ll<=t.stop_loss)or(t.direction=="short" and hh>=t.stop_loss)
                htp=(t.direction=="long" and hh>=t.take_profit)or(t.direction=="short" and ll<=t.take_profit)
                slip=SLIPPAGE_SPREAD_MULT*sp
                if htp:t.exit_price,t.exit_time,t.exit_reason=(t.take_profit-slip if t.direction=="long" else t.take_profit+slip),ts,"TP"
                elif hsl:t.exit_price,t.exit_time,t.exit_reason=(t.stop_loss-slip if t.direction=="long" else t.stop_loss+slip),ts,"SL"
            if hi==21:
                for t in[x for x in so if x.is_open]:
                    slip=SLIPPAGE_SPREAD_MULT*sp;t.exit_price,t.exit_time,t.exit_reason=(c-slip if t.direction=="long" else c+slip),ts,"TIME"
            for t in[x for x in ot if x.config_label==cfg.label and not x.is_open]:
                t.pnl=(t.exit_price-t.entry_price)*t.size if t.direction=="long" else(t.entry_price-t.exit_price)*t.size
                eq+=t.pnl;cl.append(t)
                gid=t.group_id
                if gid in pyr:pyr[gid]=[x for x in pyr[gid] if x.is_open]
                if gid in pyr and not pyr[gid]:del pyr[gid]
            ot=[t for t in ot if t.is_open]
            # Pyramid
            for gid in[g for g in pyr if g.startswith(cfg.label)]:
                layers=pyr[gid]
                if not layers or len(layers)>=PYRAMID_LAYERS:continue
                if layers[-1].profit_atr(c)>=PYRAMID_TRIGGER:
                    for t in layers:t.stop_loss=t.entry_price;t.be_moved=True
                    d=layers[0].direction;slip=SLIPPAGE_SPREAD_MULT*sp
                    ep=c+slip if d=="long" else c-slip
                    sl=ep-SL_ATR*atr if d=="long" else ep+SL_ATR*atr
                    tp=ep+TP_ATR*atr if d=="long" else ep-TP_ATR*atr
                    sz=FIXED_RISK/(SL_ATR*atr)
                    cn=sum(abs(x.size*x.entry_price) for x in ot);rem=TARGET_MAX_LEVERAGE*eq-cn
                    if rem>0:sz=min(sz,rem/ep)
                    else:sz=0
                    if sz>0 and len(ot)<MAX_POSITIONS:
                        gc+=1;nt=T(sym,d,len(layers)+1,gid,ts,ep,sl,tp,atr,sz,cfg.label)
                        ot.append(nt);layers.append(nt)
            if not(ss<=h<se):continue
            if pd.isna(avg_sp) or sp>2.0*avg_sp:continue
            pl,ps=get_poi(bar,cfg.poi_type)
            if pd.isna(pl) or pd.isna(ps):continue
            bl=pl+cfg.fract*atr if cfg.fract>0 else pl
            bs=ps-cfg.fract*atr if cfg.fract>0 else ps
            ek=(cfg.label,date)
            if ek in ent:continue
            cn=sum(abs(x.size*x.entry_price) for x in ot);rem=TARGET_MAX_LEVERAGE*eq-cn
            if cfg.direction in("long","both") and c>bl and c>bar["sma20"] and bar["sma20"]>bar["sma50"]:
                if chk(bar,cfg.filter1,"long") and len(ot)<MAX_POSITIONS:
                    slip=SLIPPAGE_SPREAD_MULT*sp;ep=c+slip
                    sl=ep-SL_ATR*atr;tp=ep+TP_ATR*atr;sz=FIXED_RISK/(SL_ATR*atr)
                    if rem>0:sz=min(sz,rem/ep)
                    else:continue
                    if sz<=0:continue
                    gc+=1;gid=f"{cfg.label}_L{gc}"
                    t=T(sym,"long",1,gid,ts,ep,sl,tp,atr,sz,cfg.label)
                    ot.append(t);pyr[gid]=[t];ent[ek]=True
            elif cfg.direction in("short","both") and c<bs and c<bar["sma20"] and bar["sma20"]<bar["sma50"]:
                if chk(bar,cfg.filter1,"short") and len(ot)<MAX_POSITIONS:
                    slip=SLIPPAGE_SPREAD_MULT*sp;ep=c-slip
                    sl=ep+SL_ATR*atr;tp=ep-TP_ATR*atr;sz=FIXED_RISK/(SL_ATR*atr)
                    if rem>0:sz=min(sz,rem/ep)
                    else:continue
                    if sz<=0:continue
                    gc+=1;gid=f"{cfg.label}_S{gc}"
                    t=T(sym,"short",1,gid,ts,ep,sl,tp,atr,sz,cfg.label)
                    ot.append(t);pyr[gid]=[t];ent[ek]=True
    for t in list(ot):
        t.exit_price=dfs[t.symbol]["close"].iloc[-1];t.exit_time=times[-1];t.exit_reason="EOD"
        t.pnl=(t.exit_price-t.entry_price)*t.size if t.direction=="long" else(t.entry_price-t.exit_price)*t.size
        eq+=t.pnl;cl.append(t)
    pnls=[t.pnl for t in cl]
    if not pnls:return 0,0,0,0
    wins=[p for p in pnls if p>0];losses=[p for p in pnls if p<=0]
    gp=sum(wins) if wins else 0;gl=abs(sum(losses)) if losses else 0
    pf=gp/gl if gl>0 else float("inf")
    return len(pnls),len(wins)/len(pnls)*100,pf,sum(pnls)/ACCOUNT_EQUITY*100

needed = sorted(set(c.symbol for c in CONFIGS))
print("Timeframe Comparison Test\n")
for tf_min, tf_name in [(15,"M15"),(30,"M30"),(60,"H1")]:
    print(f"Loading {tf_name}...")
    dfs = {}
    for sym in needed:
        df = load(sym, tf_min)
        if df is not None: dfs[sym] = df
    n,wr,pf,net = run(dfs)
    pf_s = f"{pf:.2f}" if pf < 100 else "inf"
    print(f"  {tf_name}: Trades={n:>3}, WR={wr:>5.1f}%, PF={pf_s:>6}, Net={net:>+6.1f}%\n")
