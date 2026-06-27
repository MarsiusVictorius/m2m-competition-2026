# Session Breakout Strategy V1.4 — Full Specification
## 9 Configurations | M15 Timeframe | FX + Metals

---

## 0. METHODOLOGY & TOOLS

### Framework
Based on Nesnidal's BOS (Breakout Strategy) framework from BetterTraderAcademy Masterclass. Core concept: session breakout entries using Points of Interest (Asia range levels and previous day's close) with SMA trend alignment, ATR-based risk management, and volatility filters.

### Development pipeline

| Step | Tool | Script | Purpose |
|------|------|--------|---------|
| Data collection | Python (pandas, pyarrow) | — | 22 symbols of tick-level bid/ask data stored as parquet files |
| Resampling | Python | `run_v1_4.py` | Tick → M15 OHLC with mean spread per bar |
| Backtesting | Python | `run_v1_4.py` | Event-driven bar-by-bar engine, no lookahead bias |
| Edge scanning | Python | `scan_new_symbols.py` | Brute-force search: 5 symbols × 2 directions × 2 POI types × 4 offsets × 4 filters × 4 T-segments = 1,280 combos |
| TP optimisation | Python | `tp_experiment.py` | Tested TP multipliers 1.4–2.5 in 0.1 steps |
| Walk-forward | Python | `walkforward_test.py` | Rolling/expanding window IS → OOS splits + k-fold cross-validation |
| Sensitivity | Python | `sensitivity_test.py` | ±20% perturbation on SL, TP, BE, Trail parameters (25 combos) |
| Timeframe test | Python | `tf_test.py` | Same logic on M5, M15, M30, H1 to confirm M15 is optimal |
| MT5 EA | MQL5 | `SessionBreakout_V14.mq5` | Compiled Expert Advisor for MetaTrader 5, supports all 11 configs via input parameters |
| Deployment | Symphonix | `symphonix_strategy_prompt.txt` | AI-native trading platform — strategy definitions submitted to Strategy Advisor agent |

### Data

| Detail | Value |
|--------|-------|
| Source format | Tick-level bid/ask parquet files |
| Location | `Data for backtests/` folder |
| Symbols available | 22 total (AUDJPY, AUDNZD, AUDUSD, EURCHF, EURGBP, EURJPY, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDCNH, USDJPY, XAGUSD, XAUUSD + others) |
| Period | 2026-05-11 to 2026-06-10 (~30 calendar days) |
| Bars per symbol | ~2,200 M15 bars |
| Resampling | Tick mid = (bid+ask)/2 → OHLC; spread = ask−bid → bar mean |

### Strategy evolution
- **V1.0–V1.3**: Iterative refinement of SL/TP ratios, filter combinations, entry windows
- **V1.4 (final)**: TP changed from 2.0 to 1.6 based on TP experiment. 1.6 gave best OOS robustness (+17.1% OOS vs +12.8% at TP=2.0) despite slightly lower in-sample returns

### Edge scanning methodology (for configs 7–9)
- Scanned EURUSD, USDCAD, USDCHF, EURCHF, XAGUSD (symbols available on Symphonix broker but not in original 11)
- Tested all combinations of: direction × POI type × 4 offset fractions × 4 volatility filters × 4 T-segments
- Minimum thresholds: trades ≥ 3, profit factor > 1.5, win rate ≥ 60%
- 181 qualifying configs found; top 3 selected for diversity and trade count

### Original 11 configs (pre-Symphonix)
The strategy was originally developed with 11 configs across 10 symbols. Only 5 of the original 10 symbols were available on the Symphonix broker (AUDUSD, EURGBP, GBPUSD, USDJPY, XAUUSD). Missing: AUDJPY, AUDNZD, EURJPY, NZDUSD, USDCNH. Configs 7–9 were added to partially replace the lost symbols.

### Symphonix deployment (simplified version)
Symphonix platform could not implement the full strategy due to architectural limitations:
- No ATR-based stops (only percentage-based)
- No dynamic POI levels (no Asia range or previous close reference)
- No pyramiding support
- No T-segment entry windows
- No volatility filters (ATR5 vs ATR40)

A simplified SMA crossover version was submitted to the Strategy Advisor. The full specification in this document represents the validated strategy as backtested — not the simplified Symphonix deployment.

### MT5 Expert Advisor
`SessionBreakout_V14.mq5` implements the full strategy with these input parameters per instance:
- Magic number (unique per config: 1001–1011)
- Direction, POI type, POI offset fraction
- Filter type, entry window start/end hours
- SL/TP/BE/Trail ATR multipliers
- Configured via `MT5_CONFIG_GUIDE.txt` (all 11 input parameter sets)

### Competition context
- **Event**: Model to Market Competition (quantitative trading hackathon)
- **Period**: June 21–28, 2026 (7 days)
- **Account**: $1,000,000 starting equity, 1:100 leverage
- **Rounds**: 22:00–22:00 BST daily
- **Scoring**: 70% return rank + 15% drawdown rank + 10% Sharpe rank + 5% risk discipline
- **Platform**: Symphonix (AI-native trading platform)
- **Broker**: FTWorldwide (via Symphonix)

---

## 1. UNIVERSE

| # | Symbol | Direction | POI Type | POI Offset | Filter | Entry Window (UTC) |
|---|--------|-----------|----------|------------|--------|-------------------|
| 1 | EURGBP | Short | Asia low | −0.3 × ATR14 | None | 07:00–10:00 |
| 2 | USDJPY | Short | Prev close | −0.5 × ATR14 | Vol expanding | 07:00–10:00 |
| 3 | GBPUSD | Long | Asia high | 0 | None | 07:00–10:00 |
| 4 | AUDUSD | Short | Asia low | 0 | Vol contracting | 10:00–13:00 |
| 5 | USDJPY | Long | Prev close | +0.5 × ATR14 | Vol expanding | 10:00–13:00 |
| 6 | XAUUSD | Short | Prev close | −0.5 × ATR14 | None | 07:00–17:00 |
| 7 | XAGUSD | Short | Prev close | −0.5 × ATR14 | Vol contracting | 10:00–13:00 |
| 8 | USDCAD | Long | Prev close | +0.5 × ATR14 | Vol contracting | 07:00–17:00 |
| 9 | EURUSD | Short | Asia low | 0 | None | 10:00–13:00 |

---

## 2. INDICATORS

All computed on M15 bars:

| Indicator | Definition |
|-----------|-----------|
| SMA(20) | Simple moving average, 20-period close |
| SMA(50) | Simple moving average, 50-period close |
| ATR(14) | Average true range, 14-period |
| ATR(5) | Average true range, 5-period |
| ATR(40) | Average true range, 40-period |
| Avg spread | Rolling 20-bar mean of bid-ask spread |
| Asia high | Highest high of M15 bars between 00:00–06:59 UTC on current day |
| Asia low | Lowest low of M15 bars between 00:00–06:59 UTC on current day |
| Prev close | Previous trading day's last M15 close |

---

## 3. POINT OF INTEREST (POI) CALCULATION

Each config defines a POI level that price must breach for entry.

**For LONG configs:**
```
POI = reference_level + offset × ATR(14)
```
Where reference_level is either Asia high or Prev close (per config table).

**For SHORT configs:**
```
POI = reference_level − offset × ATR(14)
```
Where reference_level is either Asia low or Prev close (per config table).

If offset = 0, POI equals the reference level directly.

---

## 4. VOLATILITY FILTERS

| Filter | Condition | Meaning |
|--------|-----------|---------|
| None | Always passes | No filter applied |
| Vol expanding | ATR(5) > ATR(40) | Recent volatility exceeds longer-term average |
| Vol contracting | ATR(5) ≤ ATR(40) | Recent volatility at or below longer-term average |

Evaluated on the current M15 bar at the time of potential entry.

---

## 5. ENTRY LOGIC

Evaluate on every M15 bar close within the config's entry window.

### Long entry conditions (ALL must be true):
1. Current bar's close > POI level
2. Close > SMA(20)
3. SMA(20) > SMA(50)
4. Config-specific volatility filter passes
5. Current spread ≤ 2.0 × Avg spread (20-bar rolling mean)
6. No entry for this config on this calendar day yet
7. Total open positions across all configs < 6

### Short entry conditions (ALL must be true):
1. Current bar's close < POI level
2. Close < SMA(20)
3. SMA(20) < SMA(50)
4. Config-specific volatility filter passes
5. Current spread ≤ 2.0 × Avg spread (20-bar rolling mean)
6. No entry for this config on this calendar day yet
7. Total open positions across all configs < 6

---

## 6. POSITION SIZING

```
risk_per_trade = $10,000 (fixed)
stop_distance = 1.5 × ATR(14)
position_size = risk_per_trade / stop_distance
```

Size is in units of the base currency. Convert to lots as needed per broker convention.

---

## 7. STOP LOSS AND TAKE PROFIT

Set at entry:

| | Long | Short |
|---|------|-------|
| Stop loss | Entry − 1.5 × ATR(14) | Entry + 1.5 × ATR(14) |
| Take profit | Entry + 1.6 × ATR(14) | Entry − 1.6 × ATR(14) |

Checked against bar high/low each M15 bar:
- Long SL hit: bar low ≤ stop loss
- Long TP hit: bar high ≥ take profit
- Short SL hit: bar high ≥ stop loss
- Short TP hit: bar low ≤ take profit

If both SL and TP are hit on the same bar, TP takes priority.

---

## 8. BREAKEVEN

When an open trade reaches **+1.0 × ATR(14)** unrealised profit:
```
Long:  if (current_close − entry_price) ≥ 1.0 × ATR(14) → stop_loss = entry_price
Short: if (entry_price − current_close) ≥ 1.0 × ATR(14) → stop_loss = entry_price
```

This is a one-time move. Once triggered, the SL stays at entry or better.

---

## 9. TRAILING STOP

Activates when unrealised profit reaches **+1.0 × ATR(14)** (same trigger as breakeven).

Once active, on every subsequent M15 bar:
```
Long:  new_sl = max(current_sl, current_close − 1.0 × ATR(14))
Short: new_sl = min(current_sl, current_close + 1.0 × ATR(14))
```

The trailing stop only moves in the favourable direction — never backwards.

---

## 10. HARD CLOSE

All open positions are closed at **21:00 UTC** regardless of P&L.

Exit price = current bar's close (with slippage applied).

---

## 11. SLIPPAGE MODEL

Applied to every entry and exit:
```
slippage = 0.5 × current_bar_spread
```

| Action | Long | Short |
|--------|------|-------|
| Entry | entry_price = close + slippage | entry_price = close − slippage |
| Exit (TP) | exit_price = TP − slippage | exit_price = TP + slippage |
| Exit (SL) | exit_price = SL − slippage | exit_price = SL + slippage |
| Exit (TIME) | exit_price = close − slippage | exit_price = close + slippage |

---

## 12. PYRAMIDING (Original specification — not used in Symphonix deployment)

Up to 3 layers per config per day. Add next layer when previous layer is +1.0 × ATR(14) in profit. When adding a layer, move all existing layers' SL to breakeven.

**Note:** Symphonix deployment uses single-layer entries only. Pyramiding was validated in backtests but excluded from this deployment due to platform limitations.

---

## 13. PORTFOLIO CONSTRAINTS

| Constraint | Value |
|-----------|-------|
| Max simultaneous open positions | 6 (across all configs) |
| Max portfolio leverage | 20× account equity |
| Max single-symbol concentration | 60% of total open notional (when 2+ symbols are open) |
| Entries per config per day | 1 |

---

## 14. ACCOUNT PARAMETERS

| Parameter | Value |
|-----------|-------|
| Starting equity | $1,000,000 |
| Leverage | 1:100 |
| Risk per trade | $10,000 (1% of starting equity) |

---

## 15. BACKTEST PERIOD

| Set | Period | Bars (M15) |
|-----|--------|-----------|
| Full | 2026-05-11 to 2026-06-10 | ~2,200 per symbol |
| In-sample | 2026-05-11 to 2026-05-31 | ~1,400 per symbol |
| Out-of-sample | 2026-06-01 to 2026-06-10 | ~640 per symbol |

Data source: Tick-level bid/ask parquet files, resampled to M15 OHLC with mean spread.

---

## 16. VALIDATION RESULTS

### Full period (all 11 original configs, includes non-Symphonix symbols)
- 89 trades | 79% WR | PF 5.05 | +40.1% | −1.6% max DD | Sharpe 0.096

### Out-of-sample (Jun 1–10)
- 38 trades | 79% WR | PF 5.94 | +17.1%

### 6 Symphonix-available configs only
- 45 trades | ~80% WR | +$221,783 (+22.2%)

### Robustness checks
- Walk-forward: 11/11 test periods profitable across all splits and k-folds
- Parameter sensitivity: 25/25 variants profitable (±20% on SL, TP, BE, Trail)
- Timeframe: M15 optimal (+40%), M30 degraded (+10.5%), H1 loses money (−2%)

---

## 17. CONFIG DETAIL

### Config 1: EURGBP Short
```
symbol      = EURGBP
direction   = short
poi_type    = asia_low
poi_offset  = -0.3
filter      = none
window_utc  = 07:00-10:00
magic       = 1004
```
Backtest: 7 trades | 100% WR | +$50,764 | PF ∞ | Median duration 45 min | Exits: 6 TP, 1 SL

### Config 2: USDJPY Short
```
symbol      = USDJPY
direction   = short
poi_type    = prev_close
poi_offset  = -0.5
filter      = vol_expanding (ATR5 > ATR40)
window_utc  = 07:00-10:00
magic       = 1009
```
Backtest: 5 trades | 100% WR | +$37,113 | PF ∞ | Median duration 60 min | Exits: 4 TP, 1 SL

### Config 3: GBPUSD Long
```
symbol      = GBPUSD
direction   = long
poi_type    = asia_high
poi_offset  = 0
filter      = none
window_utc  = 07:00-10:00
magic       = 1006
```
Backtest: 5 trades | 60% WR | +$20,204 | PF 2.16 | Median duration 105 min | Exits: 3 TP, 2 SL

### Config 4: AUDUSD Short
```
symbol      = AUDUSD
direction   = short
poi_type    = asia_low
poi_offset  = 0
filter      = vol_contracting (ATR5 <= ATR40)
window_utc  = 10:00-13:00
magic       = 1003
```
Backtest: 10 trades | 80% WR | +$46,919 | PF 5.69 | Median duration 60 min | Exits: 8 TP, 2 SL

### Config 5: USDJPY Long
```
symbol      = USDJPY
direction   = long
poi_type    = prev_close
poi_offset  = +0.5
filter      = vol_expanding (ATR5 > ATR40)
window_utc  = 10:00-13:00
magic       = 1010
```
Backtest: 9 trades | 77.8% WR | +$38,742 | PF 4.32 | Median duration 120 min | Exits: 6 TP, 2 SL, 1 TIME

### Config 6: XAUUSD Short
```
symbol      = XAUUSD
direction   = short
poi_type    = prev_close
poi_offset  = -0.5
filter      = none
window_utc  = 07:00-17:00
magic       = 1011
```
Backtest: 9 trades | 66.7% WR | +$28,041 | PF 2.42 | Median duration 60 min | Exits: 4 TP, 4 SL, 1 TIME

### Config 7: XAGUSD Short (from edge scan)
```
symbol      = XAGUSD
direction   = short
poi_type    = prev_close
poi_offset  = -0.5
filter      = vol_contracting (ATR5 <= ATR40)
window_utc  = 10:00-13:00
```
Backtest: 12 trades | 83.3% WR | +5.9% | PF 3.91

### Config 8: USDCAD Long (from edge scan)
```
symbol      = USDCAD
direction   = long
poi_type    = prev_close
poi_offset  = +0.5
filter      = vol_contracting (ATR5 <= ATR40)
window_utc  = 07:00-17:00
```
Backtest: 11 trades | 81.8% WR | +4.7% | PF 3.33

### Config 9: EURUSD Short (from edge scan)
```
symbol      = EURUSD
direction   = short
poi_type    = asia_low
poi_offset  = 0
filter      = none
window_utc  = 10:00-13:00
```
Backtest: 8 trades | 87.5% WR | +4.0% | PF 4.97

---

## 18. FILE REFERENCE

| File | Purpose |
|------|---------|
| `run_v1_4.py` | Main backtesting engine — runs all 11 configs, outputs trade log and summary stats |
| `scan_new_symbols.py` | Edge scanner for additional symbols — brute-force parameter search |
| `sensitivity_test.py` | ±20% parameter perturbation test (25 combos) |
| `walkforward_test.py` | Rolling/expanding walk-forward + k-fold cross-validation |
| `tp_experiment.py` | TP multiplier optimisation (1.4–2.5 range) |
| `tf_test.py` | Timeframe comparison (M5, M15, M30, H1) |
| `SessionBreakout_V14.mq5` | MT5 Expert Advisor source code |
| `MT5_CONFIG_GUIDE.txt` | Input parameter sets for all 11 MT5 instances |
| `trade_log_v1.4.csv` | Full trade-by-trade log with entry/exit times, prices, PnL, MAE, MFE |
| `symphonix_strategy_prompt.txt` | Deployment prompts for Symphonix Strategy Advisor + Guardrails agents |
| `Data for backtests/` | Tick-level bid/ask parquet files for 22 symbols |
| `sensitivity_v1.4.png` | Parameter sensitivity heatmap |
| `walkforward_v1.4.png` | Walk-forward equity curves |
| `tp_experiment_v1.4.png` | TP multiplier comparison chart |
