# SESSION BREAKOUT STRATEGY V1.4 — MANUAL TRADING GUIDE
## M2M Competition | $1M Account | June 21-28, 2026

---

## GLOBAL RULES (Apply to ALL configs)

| Parameter | Value |
|-----------|-------|
| Timeframe | M15 |
| ATR period | 14 bars |
| Trend filter | SMA(20) vs SMA(50) alignment |
| Stop Loss | 1.5 × ATR(14) from entry |
| Take Profit | 1.6 × ATR(14) from entry |
| Breakeven | Move SL to entry price when +1.0 × ATR(14) in profit |
| Trailing Stop | Activate at +1.0 × ATR in profit, trail at 1.0 × ATR distance |
| Risk per trade | $10,000 fixed |
| Position size | $10,000 ÷ (1.5 × ATR14) = lots |
| Max positions | 6 across ALL configs |
| Hard close | 22:00 BST (21:00 UTC) — close everything, no exceptions |
| Spread filter | Skip if spread > 2× the 20-bar average spread |
| Entries per day | 1 per config maximum |

---

## MORNING ROUTINE (Do this at 07:45 BST every day)

1. Open each symbol on M15
2. Draw the **Asia range** — highest high and lowest low from 01:00-07:59 BST candles
3. Note **yesterday's close** price for prev_close configs
4. Eyeball **ATR(14)** — average range of last 14 M15 candles
5. Check **SMA alignment** — are both moving averages sloping with your trade direction?
6. Check **volatility** — are recent candles bigger or smaller than older ones? (for vol filter configs)

---

## CONFIG 1: EURGBP SHORT ⭐ Best performer

**Market bias:** Bearish. Looking for GBP strength / EUR weakness during London open.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | EURGBP |
| Direction | **SELL** |
| Entry window | **08:00 - 11:00 BST** (London open, T1) |
| POI level | Asia range LOW − 0.3 × ATR(14) |
| Volatility filter | None |
| Trend filter | Price < SMA(20), SMA(20) < SMA(50) |

### Entry checklist
1. It's between 08:00-11:00 BST
2. Mark Asia range low (lowest price 01:00-07:59)
3. Calculate POI = Asia low − (0.3 × ATR14)
4. ✅ Price is BELOW the POI level
5. ✅ Price is BELOW SMA(20)
6. ✅ SMA(20) is BELOW SMA(50)
7. ✅ Spread looks normal (not spiking)
8. → **SELL at market**

### Exit rules
- **SL** = Entry price + 1.5 × ATR(14)
- **TP** = Entry price − 1.6 × ATR(14)
- When trade is +1.0 × ATR in profit → move SL to entry (breakeven)
- When trade is +1.0 × ATR in profit → start trailing SL at 1.0 × ATR below price
- Hard close at 22:00 BST if still open

### Backtest stats (May 11 - Jun 10, 2026)
| Metric | Value |
|--------|-------|
| Trades | 7 |
| Win rate | **100%** (7/7) |
| Net PnL | +$50,764 |
| Avg trade | +$7,252 |
| Median duration | **45 min** (fastest config) |
| Range | 15 min - 2 hours |
| Exits | 6 TP, 1 SL (still profitable due to trailing) |
| Avg adverse excursion | -$861 (very low heat) |
| Entry hours (UTC) | 07:00, 08:00, 09:00 |

### Notes
- This is your sniper — fast in, fast out
- Typically hits TP in 1-3 M15 candles
- Very low drawdown per trade
- The 0.3 ATR offset below Asia low means you're entering AFTER the breakdown is confirmed

---

## CONFIG 2: USDJPY SHORT

**Market bias:** Bearish yen pairs. Looking for JPY strength / USD weakness during volatile London conditions.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | USDJPY |
| Direction | **SELL** |
| Entry window | **08:00 - 11:00 BST** (T1) |
| POI level | Previous day's close − 0.5 × ATR(14) |
| Volatility filter | **Expanding** — ATR(5) > ATR(40) |
| Trend filter | Price < SMA(20), SMA(20) < SMA(50) |

### Entry checklist
1. It's between 08:00-11:00 BST
2. Note yesterday's close price
3. Calculate POI = prev close − (0.5 × ATR14)
4. ✅ Price is BELOW the POI level
5. ✅ Price is BELOW SMA(20)
6. ✅ SMA(20) is BELOW SMA(50)
7. ✅ Volatility is EXPANDING — recent candles are BIGGER than average
8. ✅ Spread looks normal
9. → **SELL at market**

### Exit rules
- **SL** = Entry + 1.5 × ATR(14)
- **TP** = Entry − 1.6 × ATR(14)
- Breakeven at +1.0 ATR, trail at 1.0 ATR

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 5 |
| Win rate | **100%** (5/5) |
| Net PnL | +$37,113 |
| Avg trade | +$7,423 |
| Median duration | 60 min |
| Range | 30 min - 4.5 hours |
| Exits | 4 TP, 1 SL (profitable due to trailing) |
| Avg adverse excursion | -$3,346 |
| Entry hours (UTC) | 07:00, 08:00 |

### Notes
- Only fires on volatile days — the vol expanding filter screens out quiet sessions
- When it fires, it has very high conviction
- The 0.5 ATR offset means you wait for a solid break below prev close
- JPY moves can be fast — watch for the breakeven trigger

---

## CONFIG 3: GBPUSD LONG

**Market bias:** Bullish. Looking for GBP strength during London open.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | GBPUSD |
| Direction | **BUY** |
| Entry window | **08:00 - 11:00 BST** (T1) |
| POI level | Asia range HIGH (no offset) |
| Volatility filter | None |
| Trend filter | Price > SMA(20), SMA(20) > SMA(50) |

### Entry checklist
1. It's between 08:00-11:00 BST
2. Mark Asia range high (highest price 01:00-07:59)
3. ✅ Price has broken ABOVE Asia high
4. ✅ Price is ABOVE SMA(20)
5. ✅ SMA(20) is ABOVE SMA(50)
6. ✅ Spread looks normal
7. → **BUY at market**

### Exit rules
- **SL** = Entry − 1.5 × ATR(14)
- **TP** = Entry + 1.6 × ATR(14)
- Breakeven at +1.0 ATR, trail at 1.0 ATR

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 5 |
| Win rate | 60% (3/5) |
| Net PnL | +$20,204 |
| Avg trade | +$4,041 |
| Median duration | 105 min |
| Range | 15 min - 2 hours |
| Exits | 3 TP, 2 SL |
| Avg adverse excursion | -$2,504 |
| Entry hours (UTC) | 07:00, 08:00, 09:00 |

### Notes
- Lower WR than others but still positive expectancy
- The Asia high breakout is a classic London session play
- Slower trades — expect 1-2 hours to resolve
- No filter means more entries but some will fail

---

## CONFIG 4: AUDUSD SHORT

**Market bias:** Bearish. Looking for AUD weakness during London/NY overlap.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | AUDUSD |
| Direction | **SELL** |
| Entry window | **12:00 - 15:00 BST** (T2, NY crossover) |
| POI level | Asia range LOW (no offset) |
| Volatility filter | **Contracting** — ATR(5) ≤ ATR(40) |
| Trend filter | Price < SMA(20), SMA(20) < SMA(50) |

### Entry checklist
1. It's between 12:00-15:00 BST
2. Mark Asia range low from earlier (01:00-07:59)
3. ✅ Price has broken BELOW Asia low
4. ✅ Price is BELOW SMA(20)
5. ✅ SMA(20) is BELOW SMA(50)
6. ✅ Volatility is CONTRACTING — recent candles are SMALLER than average
7. ✅ Spread looks normal
8. → **SELL at market**

### Exit rules
- **SL** = Entry + 1.5 × ATR(14)
- **TP** = Entry − 1.6 × ATR(14)
- Breakeven at +1.0 ATR, trail at 1.0 ATR

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 10 |
| Win rate | **80%** (8/10) |
| Net PnL | +$46,919 |
| Avg trade | +$4,692 |
| Median duration | 60 min |
| Range | 30 min - 2.25 hours |
| Exits | 8 TP, 2 SL |
| Avg adverse excursion | -$2,867 |
| Entry hours (UTC) | 10:00, 11:00, 12:00 |

### Notes
- Vol contracting filter = quiet buildup before the breakdown
- This is the "calm before the storm" pattern — price coils tight then breaks
- Afternoon session entry — check levels in morning, wait for NY overlap
- Strong PF of 5.69 — losses are small, wins are full

---

## CONFIG 5: USDJPY LONG

**Market bias:** Bullish USD/JPY. Looking for USD strength during NY crossover.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | USDJPY |
| Direction | **BUY** |
| Entry window | **11:00 - 14:00 BST** (T2) |
| POI level | Previous day's close + 0.5 × ATR(14) |
| Volatility filter | **Expanding** — ATR(5) > ATR(40) |
| Trend filter | Price > SMA(20), SMA(20) > SMA(50) |

### Entry checklist
1. It's between 11:00-14:00 BST
2. Note yesterday's close price
3. Calculate POI = prev close + (0.5 × ATR14)
4. ✅ Price is ABOVE the POI level
5. ✅ Price is ABOVE SMA(20)
6. ✅ SMA(20) is ABOVE SMA(50)
7. ✅ Volatility is EXPANDING — recent candles are BIGGER than average
8. ✅ Spread looks normal
9. → **BUY at market**

### Exit rules
- **SL** = Entry − 1.5 × ATR(14)
- **TP** = Entry + 1.6 × ATR(14)
- Breakeven at +1.0 ATR, trail at 1.0 ATR

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 9 |
| Win rate | **77.8%** (7/9) |
| Net PnL | +$38,742 |
| Avg trade | +$4,305 |
| Median duration | **120 min** (slowest config) |
| Range | 15 min - 7.75 hours |
| Exits | 6 TP, 2 SL, 1 TIME (hard close) |
| Avg adverse excursion | -$1,519 (low) |
| Entry hours (UTC) | 10:00-12:00, sometimes 17:00 |

### Notes
- Mirror of Config 2 but opposite direction and later window
- Slowest config — some trades run 4+ hours
- The TIME exit means one trade was still running at hard close
- Vol expanding filter ensures you're trading momentum days only
- Low MAE means it doesn't go against you much before working

---

## CONFIG 6: XAUUSD SHORT (Gold)

**Market bias:** Bearish gold. Looking for gold weakness during active sessions.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | XAUUSD |
| Direction | **SELL** |
| Entry window | **08:00 - 18:00 BST** (full day) |
| POI level | Previous day's close − 0.5 × ATR(14) |
| Volatility filter | None |
| Trend filter | Price < SMA(20), SMA(20) < SMA(50) |

### Entry checklist
1. It's between 08:00-18:00 BST (wide window)
2. Note yesterday's close for gold
3. Calculate POI = prev close − (0.5 × ATR14)
4. ✅ Price is BELOW the POI level
5. ✅ Price is BELOW SMA(20)
6. ✅ SMA(20) is BELOW SMA(50)
7. ✅ Spread looks normal (gold spreads can be wide — check)
8. → **SELL at market**

### Exit rules
- **SL** = Entry + 1.5 × ATR(14)
- **TP** = Entry − 1.6 × ATR(14)
- Breakeven at +1.0 ATR, trail at 1.0 ATR

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 9 |
| Win rate | 66.7% (6/9) |
| Net PnL | +$28,041 |
| Avg trade | +$3,116 |
| Median duration | 60 min |
| Range | 15 min - 4.75 hours |
| Exits | 4 TP, 4 SL, 1 TIME |
| Avg adverse excursion | -$4,167 (highest of all configs) |
| Entry hours (UTC) | 07:00-08:00, 13:00-16:00 |

### Notes
- Widest entry window — can fire any time during the day
- Lowest WR of the 6 configs but still profitable
- Gold has the highest adverse excursion — expect some heat
- Gold spreads spike around news — be extra careful with spread filter
- Entries cluster in two groups: London open and NY afternoon

---

## BONUS CONFIGS (New edges from symbol scan)

These 3 configs were found by scanning symbols available on the broker that weren't in the original 11. Less trade count so lower confidence, but passed all filters.

---

## CONFIG 7: XAGUSD SHORT (Silver)

**Market bias:** Bearish silver during NY crossover with quiet volatility.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | XAGUSD |
| Direction | **SELL** |
| Entry window | **11:00 - 14:00 BST** (T2) |
| POI level | Previous day's close − 0.5 × ATR(14) |
| Volatility filter | **Contracting** — ATR(5) ≤ ATR(40) |
| Trend filter | Price < SMA(20), SMA(20) < SMA(50) |

### Entry checklist
1. It's between 11:00-14:00 BST
2. Note yesterday's close for silver
3. Calculate POI = prev close − (0.5 × ATR14)
4. ✅ Price is BELOW the POI level
5. ✅ Price is BELOW SMA(20), SMA(20) < SMA(50)
6. ✅ Volatility is CONTRACTING
7. ✅ Spread looks normal
8. → **SELL at market**

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 12 |
| Win rate | **83.3%** |
| Profit factor | 3.91 |
| Net return | +5.9% |

---

## CONFIG 8: USDCAD LONG

**Market bias:** Bullish USD/CAD during quiet conditions.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | USDCAD |
| Direction | **BUY** |
| Entry window | **08:00 - 18:00 BST** (full day) |
| POI level | Previous day's close + 0.5 × ATR(14) |
| Volatility filter | **Contracting** — ATR(5) ≤ ATR(40) |
| Trend filter | Price > SMA(20), SMA(20) > SMA(50) |

### Entry checklist
1. It's between 08:00-18:00 BST
2. Note yesterday's close for USDCAD
3. Calculate POI = prev close + (0.5 × ATR14)
4. ✅ Price is ABOVE the POI level
5. ✅ Price is ABOVE SMA(20), SMA(20) > SMA(50)
6. ✅ Volatility is CONTRACTING
7. ✅ Spread looks normal
8. → **BUY at market**

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 11 |
| Win rate | **81.8%** |
| Profit factor | 3.33 |
| Net return | +4.7% |

---

## CONFIG 9: EURUSD SHORT

**Market bias:** Bearish EUR during NY crossover.

### Setup
| Detail | Value |
|--------|-------|
| Symbol | EURUSD |
| Direction | **SELL** |
| Entry window | **11:00 - 14:00 BST** (T2) |
| POI level | Asia range LOW (no offset) |
| Volatility filter | None |
| Trend filter | Price < SMA(20), SMA(20) < SMA(50) |

### Entry checklist
1. It's between 11:00-14:00 BST
2. Mark Asia range low from 01:00-07:59
3. ✅ Price has broken BELOW Asia low
4. ✅ Price is BELOW SMA(20), SMA(20) < SMA(50)
5. ✅ Spread looks normal
6. → **SELL at market**

### Backtest stats
| Metric | Value |
|--------|-------|
| Trades | 8 |
| Win rate | **87.5%** |
| Profit factor | 4.97 |
| Net return | +4.0% |

---

## DAILY SCHEDULE (BST)

```
07:45  Morning prep — mark Asia ranges, note prev closes, check SMA alignment
08:00  WINDOW OPENS: EURGBP short, USDJPY short, GBPUSD long, XAUUSD short, USDCAD long
11:00  WINDOW OPENS: AUDUSD short, USDJPY long, XAGUSD short, EURUSD short
       WINDOW CLOSES: EURGBP short, USDJPY short, GBPUSD long
14:00  WINDOW CLOSES: AUDUSD short, USDJPY long, XAGUSD short, EURUSD short
15:00  Manage open positions only — no new entries (except XAUUSD, USDCAD)
18:00  WINDOW CLOSES: XAUUSD short, USDCAD long
22:00  HARD CLOSE — close every open position, no exceptions
```

## POSITION SIZING QUICK REFERENCE

**Formula:** Lots = $10,000 ÷ (1.5 × ATR14 × pip value)

Rough guide (actual ATR varies daily):

| Symbol | Typical ATR14 (M15) | Approx lot size |
|--------|---------------------|-----------------|
| EURGBP | ~0.00033 | ~20M units |
| GBPUSD | ~0.00065 | ~10M units |
| AUDUSD | ~0.00045 | ~15M units |
| USDJPY | ~0.11 | ~60K units |
| XAUUSD | ~5.5 | ~1,200 units |
| XAGUSD | ~0.07 | ~95K units |
| USDCAD | ~0.00055 | ~12M units |
| EURUSD | ~0.00050 | ~13M units |

⚠️ Always recalculate from live ATR — these are approximations only.

## RULES FOR THE COMPETITION

1. **ONLY trade these 9 configs.** Nothing else. No "I feel like gold is going up."
2. **Follow every step on the checklist.** If ANY checkbox fails, DO NOT ENTER.
3. **One entry per config per day.** Missed it? Wait for tomorrow.
4. **Max 6 positions at once.** If you already have 6 open, wait for one to close.
5. **Hard close at 22:00 BST.** Set an alarm.
6. **After entry: set SL and TP immediately.** Don't "watch it for a bit."
7. **Move to breakeven at +1.0 ATR.** This protects your capital.
8. **No revenge trading.** If you take 2 losses, walk away and wait for next window.
9. **Record every trade.** Symbol, time, entry, SL, TP, exit, reason.

---

## PORTFOLIO SUMMARY

| Config | Symbol | Direction | Window (BST) | WR | Net PnL |
|--------|--------|-----------|-------------|-----|---------|
| 1 | EURGBP | SHORT | 08-11 | 100% | +$50,764 |
| 2 | USDJPY | SHORT | 08-11 | 100% | +$37,113 |
| 3 | GBPUSD | LONG | 08-11 | 60% | +$20,204 |
| 4 | AUDUSD | SHORT | 12-15 | 80% | +$46,919 |
| 5 | USDJPY | LONG | 11-14 | 78% | +$38,742 |
| 6 | XAUUSD | SHORT | 08-18 | 67% | +$28,041 |
| 7 | XAGUSD | SHORT | 11-14 | 83% | scan |
| 8 | USDCAD | LONG | 08-18 | 82% | scan |
| 9 | EURUSD | SHORT | 11-14 | 88% | scan |
| **Total (6 core)** | | | | **~80%** | **+$221,783** |

Combined backtest: 45 trades, ~80% WR, +22% return on $1M in 30 days.

---

*Generated from V1.4 backtests (May 11 - Jun 10, 2026). Past performance does not guarantee future results. Strategy validated with walk-forward analysis, parameter sensitivity, and out-of-sample testing.*
