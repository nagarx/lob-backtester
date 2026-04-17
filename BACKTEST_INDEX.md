# Backtest Index

**Living ledger of all backtest experiments.** Updated after every run.

---

## Round 3: IBKR-Validated + BSM Theta (2026-03-14)

Cost model: IBKR 318-fill empirical commission ($0.70/contract) + BSM theta (replaces broken 10 bps/min constant). Theta was 22-78x overcalibrated in Round 2; this round has correct BSM-based theta.

### 0DTE Option P&L (IBKR+BSM Calibrated)

| Run | Option Return | Option Win Rate | Avg P&L/Trade | Spread | Commission | Theta (BSM) | Total Cost | Hold (min) | Move (bps) |
|---|---|---|---|---|---|---|---|---|---|
| ibkr_no_hold | **-43.0%** | 20.07% | -$3.89 | $2.39 | $1.40 | $0.04 | $3.84 | 0.1 | -0.05 |
| ibkr_h10_hold | **-16.6%** | 38.14% | -$4.57 | $2.39 | $1.40 | $0.42 | $4.21 | 1.0 | -0.41 |
| ibkr_h60_hold | **-3.7%** | 43.71% | -$4.65 | $2.41 | $1.40 | $2.54 | $6.35 | 6.0 | +1.82 |
| ibkr_h300_hold | **-3.3%** | 40.12% | -$19.88 | $2.34 | $1.39 | $12.64 | $16.37 | 29.8 | -3.61 |
| ibkr_reversal | **-32.0%** | 21.16% | -$4.03 | $2.39 | $1.40 | $0.11 | $3.90 | 0.3 | -0.14 |

### Theta Correction Impact (Round 2 → Round 3)

| Run | Round 2 Theta | Round 3 Theta (BSM) | Round 2 Option Return | Round 3 Option Return | Improvement |
|---|---|---|---|---|---|
| ibkr_no_hold | $1.83 | **$0.04** | -62.3% | **-43.0%** | +19.3pp |
| ibkr_h10_hold | $18.26 | **$0.42** | -81.3% | **-16.6%** | +64.7pp |
| ibkr_h60_hold | $109.36 | **$2.54** | -87.7% | **-3.7%** | +84.0pp |
| ibkr_h300_hold | $544.06 | **$12.64** | -92.1% | **-3.3%** | +88.8pp |
| ibkr_reversal | $4.94 | **$0.11** | -70.1% | **-32.0%** | +38.1pp |

The theta fix reveals that **H60 and H300 are close to breakeven** (-3.7% and -3.3%) — far better than the -87%/-92% reported in Round 2. The dominant remaining cost is spread+commission ($3.80/trade), not theta.

### IBKR-Validated Cost Model

Source: `IBKR-transactions-trades/IBKR_REAL_WORLD_TRADING_REPORT.md` (318 real fills)

| Component | Call | Put | Source |
|---|---|---|---|
| Half-spread (per share) | $0.015 | $0.010 | OPRA median |
| Full spread (per contract) | $3.00 | $2.00 | x 100 shares |
| Commission (round-trip) | $1.40 | $1.40 | IBKR 318-fill median $0.70/leg |
| **Round-trip (excl theta)** | **$4.40** | **$3.40** | per contract |
| Theta (BSM, 1 min hold) | $0.42 | $0.42 | BSM at 14:00, IV=40% |
| Theta (BSM, 6 min hold) | $2.54 | $2.54 | BSM at 14:00, IV=40% |
| Theta (BSM, 30 min hold) | $12.64 | $12.64 | BSM at 14:00, IV=40% |

ATM premium: call $1.88 median (OPRA, validated by IBKR $1.86), put $1.31 median. Delta: 0.50.

### Breakeven Analysis (IBKR-validated)

| Scenario | RT Cost | Breakeven (bps on $180) |
|---|---|---|
| ATM 0DTE Call, 1-min hold | $4.82 | 5.4 bps |
| ATM 0DTE Put, 1-min hold | $3.82 | 4.2 bps |
| ATM 0DTE Call, 6-min hold | $6.94 | 7.7 bps |
| Deep ITM (delta=0.95), no theta | $2.40 | 1.4 bps |

---

## Move Magnitude Analysis

At readability-gated windows (14,497 samples, agree=1.0, conf>0.65, spread<=1.05):

| Horizon | Dir Move (mean) | Dir Move (median) | Abs Move (mean) | Win > 0 |
|---|---|---|---|---|
| 10 events | -0.06 bps | 0.00 bps | 3.9 bps | 47.2% |
| 60 events | -0.07 bps | 0.00 bps | 9.5 bps | 49.0% |
| 300 events | +0.03 bps | 0.00 bps | 20.8 bps | 49.7% |

The market moves are large enough (9.5 bps mean at H60 > 5.4 bps breakeven). The problem is the model has **no directional edge** — the predicted direction is uncorrelated with actual price movement.

---

## Round 2: OPRA-Calibrated (DEPRECATED — theta was 22-78x too high)

**Note:** Round 2 used a constant 10 bps/min theta model that was 22-78x too high. Results are kept for reference but should not be used for decision-making.

| Run | Option Return (R2) | Option Return (R3 corrected) | Theta Error |
|---|---|---|---|
| H10 hold | -81.3% | -16.6% | 43x too high |
| H60 hold | -87.7% | -3.7% | 43x too high |
| H300 hold | -92.1% | -3.3% | 43x too high |

---

## Round 1: Equity-Only Baseline (pre-OPRA)

| Run | Holding | Trades | Return | MaxDD | Win Rate | Expectancy |
|---|---|---|---|---|---|---|
| no_hold (baseline) | none (flicker) | 14,051 | -36.79% | 36.90% | 29.26% | -$2.63 |
| h10_hold | horizon_aligned_10 | 7,274 | -22.35% | 22.40% | 41.55% | -$3.25 |
| h60_hold | horizon_aligned_60 | 1,573 | -3.68% | 3.96% | 47.78% | -$1.46 |
| h300_hold | horizon_aligned_300 | 333 | -1.70% | 2.12% | 49.70% | -$6.92 |
| reversal | direction_reversal_300 | 15,903 | -41.20% | 41.22% | 24.74% | -$2.65 |

---

## Root Cause Analysis

### The signal-cost mismatch

The TLOB labeling strategy classifies moves with a ±2 bps threshold. The model achieves 95.50% accuracy on these labels. But trading costs require 4.2-5.4 bps of directional movement per trade. The model perfectly predicts sub-threshold moves that cannot cover costs.

### What the corrected cost model reveals

With BSM theta (Round 3), the picture is very different from the broken Round 2:

1. **Theta is NOT the dominant cost** for short holds. At H10 (1 min), theta is only $0.42 vs $3.79 in spread+commission. The spread is the dominant cost.

2. **H60 and H300 are nearly breakeven** (-3.7% and -3.3%). The directional edge at H60 is +1.82 bps mean — not zero, and in the right direction. With better signal quality, H60 could be profitable.

3. **The path to profitability is clear**: increase the labeling threshold to match the breakeven cost, so the model only predicts moves large enough to trade profitably. The profiler recommends ±12 bps at 1-minute (H60), which would create a ~31/38/31 class balance where every directional label represents a profitable move.

---

## Lessons Learned (Updated)

1. **Validate cost models against real data.** The 10 bps/min theta was 43x too high — discovered only when compared to BSM and IBKR screenshots. Always cross-check with first principles and real fills.

2. **Commission is the dominant fixed cost** for short holds. $1.40 round-trip from IBKR (validated) vs $0.04-$0.42 theta for 0-1 min holds.

3. **Label threshold must match trading costs.** TLOB ±2 bps labels are 2-3x below the 4.2-5.4 bps breakeven. Profit-threshold labeling (±12 bps at H60) aligns labels with tradeable moves.

4. **H60 is the sweet spot.** Absolute moves (9.5 bps mean) exceed breakeven (5.4 bps), and the Round 3 H60 result (-3.7%) is nearly breakeven even without signal-aligned labels. With profit-threshold labels, this horizon has the best chance of profitability.

5. **Deep ITM calls have ~4x lower breakeven** (1.4 bps vs 5.4 bps). This opens a parallel strategy path worth investigating.

---

## Round 4: TLOB Regression Backtests (2026-03-15)

Model: TLOB 128-feat regression (R²=0.464, IC=0.677, DA=74.9%). Predicts continuous bps returns at H10.
Strategy: `RegressionStrategy` — entry gate: |predicted_return| > threshold AND spread <= 1.05 bps.
Signal source: `lob-model-trainer/outputs/experiments/nvda_tlob_128feat_regression_h10/signals/test/`

### H10 Hold (10 events, ~1 second)

| Threshold | Trades | Option Return | Notes |
|-----------|--------|---------------|-------|
| 0.7 bps (deep ITM) | 4,270 | -19.75% | Too many trades, costs dominate |
| 2.0 bps (ITM) | 3,900 | -19.07% | |
| 3.0 bps (ITM) | 3,420 | -15.78% | |
| 5.0 bps (ATM) | 1,799 | -7.53% | |
| 8.0 bps (high conviction) | 214 | -0.93% | Approaching breakeven |
| 10.0 bps (very high) | 54 | -0.35% | Near breakeven, very few trades |

### H60 Hold (60 events, ~6 seconds)

| Threshold | Trades | Option Return | Notes |
|-----------|--------|---------------|-------|
| 0.7 bps (deep ITM) | 816 | -3.99% | |
| 3.0 bps (ITM) | 775 | -2.71% | Best return in this sweep |
| 5.0 bps (ATM) | 637 | -3.66% | |
| 8.0 bps (high conviction) | 151 | -0.86% | |
| 10.0 bps (very high) | 45 | -0.77% | |

### Key Finding: Label-Execution Mismatch

Model was trained on TLOB **smoothed-average** labels (mean of next 10 mid-price changes) but backtest executes **point-to-point** (price at exit minus price at entry). This mismatch causes the model's 74.9% directional accuracy to translate to only ~38% execution win rate. The next experiment should use `return_type = "point_return"` labels to align training with execution.

---

## Round 5: Readability Hybrid Backtest (2026-03-16)

Strategy: `ReadabilityHybridStrategy` -- dual gate combining HMHP classification readability with Ridge regression magnitude filtering.
Direction source: HMHP 40-feat classification predictions (95.50% DA at full readability gate).
Magnitude source: TemporalRidge regression (IC=0.616, 54 params).
Signal source: `lob-model-trainer/outputs/experiments/hybrid_readability_ridge_h10/signals/test/`
Samples: 50,724 (identical prices across both models, verified with `np.allclose`).

### H10 Hold (10 events, ~1 second)

| Agreement | Confidence | Min Return | Trades | Option Return | Win Rate |
|-----------|------------|------------|--------|---------------|----------|
| 1.0 | >0.50 | 1 bps | 4,048 | -20.10% | 36.0% |
| 1.0 | >0.50 | 3 bps | 3,639 | -17.12% | 37.3% |
| 1.0 | >0.50 | 5 bps | 2,557 | -10.91% | 39.2% |
| 1.0 | >0.50 | 8 bps | 842 | -3.41% | 41.1% |
| 1.0 | >0.65 | 1 bps | 3,592 | -16.59% | 37.8% |
| 1.0 | >0.65 | 3 bps | 3,303 | -15.28% | 38.2% |
| 1.0 | >0.65 | 5 bps | 2,395 | -10.62% | 39.4% |
| 1.0 | >0.65 | 8 bps | 804 | -3.03% | 41.9% |
| 1.0 | >0.80 | any | 0 | 0.00% | N/A |

### H60 Hold (60 events, ~6 seconds)

| Agreement | Confidence | Min Return | Trades | Option Return | Win Rate |
|-----------|------------|------------|--------|---------------|----------|
| 1.0 | >0.50 | 1 bps | 807 | -5.77% | 40.1% |
| 1.0 | >0.50 | 3 bps | 788 | -6.51% | 39.6% |
| 1.0 | >0.50 | 5 bps | 714 | -3.11% | 42.7% |
| 1.0 | >0.50 | 8 bps | 401 | -4.17% | 40.2% |
| 1.0 | >0.65 | 1 bps | 786 | -2.97% | 42.9% |
| 1.0 | >0.65 | 3 bps | 770 | -5.19% | 40.5% |
| 1.0 | >0.65 | 5 bps | 701 | **-2.67%** | 42.8% |
| 1.0 | >0.65 | 8 bps | 397 | -4.34% | 39.8% |
| 1.0 | >0.80 | any | 0 | 0.00% | N/A |

### Comparison: Hybrid vs Individual Strategies

| Strategy | Best Config | Trades | Option Return |
|---|---|---|---|
| **Hybrid (readability + magnitude)** | agree=1.0, conf>0.65, \|ret\|>=5bps, h=60 | 701 | **-2.67%** |
| Pure Ridge regression | \|ret\|>=10bps, h=10 | 333 | **-1.14%** |
| Pure TLOB regression | \|ret\|>=10bps, h=10 | 54 | -0.35% |
| Pure Readability (HMHP) | h60 hold | 1,573 | -3.70% |

### Key Finding: Readability Gate Is Not Additive

The hybrid strategy (-2.67% best) performs WORSE than pure Ridge regression at 10 bps threshold (-1.14%). The readability gate from classification does not add value on top of the regression magnitude filter because:

1. **Both models predict smoothed-average returns**, not point-to-point tradeable returns. However, **P0 validation (2026-03-17) showed the label-to-label correlation is r=0.642 with 69.3% directional win rate** — the mismatch is smaller than originally believed. The primary performance bottleneck is **cost structure** (ATM breakeven 5.4 bps > mean return 2.65 bps), not label misalignment. See `lob-model-trainer/reports/p0_label_execution_validation_2026_03_17.md`.

2. **The readability gate increases trade count**: At the hybrid's best config (701 trades), the strategy takes more trades than pure Ridge at 10bps (333 trades) because the 5 bps magnitude threshold is lower. More trades at lower conviction = worse performance.

3. **The confirmation score ceiling (0.667) prevents high-conviction filtering**: All confidence > 0.80 configurations produce zero trades. The HMHP 40-feat model's confirmation mechanism saturates below the planned threshold.

4. **Agreement is binary, not graduated**: agreement=0.9 and agreement=1.0 produce identical results because the HMHP agreement distribution is bimodal (1.0 or much lower).

---

## Config Archive

All configs stored in `outputs/backtests/{run_id}/config.yaml` with full reproducibility.

### Signal Sources by Round

| Round | Signal Source | Model | Samples |
|-------|-------------|-------|---------|
| R1-R3 | `lob-model-trainer/outputs/experiments/nvda_hmhp_40feat_h10/signals/test/` | HMHP 40-feat classification | 50,724 |
| R4 | `lob-model-trainer/outputs/experiments/nvda_tlob_128feat_regression_h10/signals/test/` | TLOB 128-feat regression | 50,724 |
| R5 | `lob-model-trainer/outputs/experiments/hybrid_readability_ridge_h10/signals/test/` | HMHP + Ridge merged | 50,724 |

| R6 | `lob-model-trainer/outputs/experiments/e4_tlob_h60/signals/test/` | TLOB E4 time-based H60 | 218,163 |
| R7 | `lob-model-trainer/outputs/experiments/e5_60s_huber_nocvml/signals/test/` | TLOB E5 time-based 60s H10 | 8,337 |
| R8 | `lob-model-trainer/outputs/experiments/e6_calibrated_conviction/signals/test/` | TLOB E6 calibrated | 8,337 |

### Calibration Sources

- OPRA: `opra-statistical-profiler/output_opra_nvda/` (8-day NVDA options)
- IBKR: `IBKR-transactions-trades/IBKR_REAL_WORLD_TRADING_REPORT.md` (318 real fills)
- IBKR Cost Audit: `IBKR-transactions-trades/COST_AUDIT_2026_03.md` (316 fills, corrected breakevens)
- Data R1-R5: XNAS 128-feat test split (50,724 samples, 35 days)
- Data R6: XNAS 98-feat time-based E4 test split (218,163 samples, 35 days)

---

## Round 6: E4 TLOB Time-Based H60 (2026-03-18)

Model: TLOB 2L/32H/2Heads (92,690 params), trained on E4 time-based 5-second export. First model on time-based sampled data. Test IC=0.136, R2=0.015, DA=0.544. Holding: 60 events = 5 minutes.

Cost model: IBKR validated ($0.70/contract commission). ATM: half-spread=$0.015, delta=0.50, breakeven=4.9 bps. Deep ITM: half-spread=$0.005, delta=0.95, breakeven=1.4 bps.

### ATM Options (delta=0.50)

| Threshold | Trades | 0DTE Return | Win Rate | Avg P&L/Trade |
|-----------|--------|-------------|----------|---------------|
| 0.7 bps | 3,145 | -19.81% | 27.4% | -$6.30 |
| 2.0 bps | 2,488 | -15.06% | 32.4% | -$6.05 |
| 3.0 bps | 2,153 | -13.68% | 32.2% | -$6.35 |
| 5.0 bps | 763 | -5.25% | 36.3% | -$6.88 |

### Deep ITM Options (delta=0.95)

| Threshold | Trades | 0DTE Return | Win Rate | Avg P&L/Trade |
|-----------|--------|-------------|----------|---------------|
| 0.7 bps | 3,145 | -14.24% | 38.0% | -$4.53 |
| 2.0 bps | 2,488 | -10.71% | 41.7% | -$4.30 |
| 3.0 bps | 2,153 | -10.73% | 41.5% | -$4.98 |
| 5.0 bps | 763 | **-3.68%** | **45.0%** | -$4.82 |

### Key Finding

Deep ITM consistently better than ATM (+5-6pp return, +9-11pp win rate). Best result: Deep ITM at 5 bps threshold (-3.68%, 45% win rate). Still negative — IC=0.136 is insufficient for profitability. Model direction accuracy (38-45%) is below the ~50% needed to overcome costs.

### Comparison with Prior Rounds

| Round | Model | Best Option Return | Best Win Rate |
|-------|-------|--------------------|---------------|
| R3 (H60) | HMHP classification | -3.7% | 43.7% |
| R5 | HMHP+Ridge hybrid | -2.67% | 42.8% |
| **R6 (Deep ITM)** | **TLOB E4 time-based** | **-3.68%** | **45.0%** |

E4 achieves the highest win rate (45.0%) in pipeline history, but returns remain negative.

---

## Round 7: E5 Time-Bin Sweep H10=10min (2026-03-19)

Model: TLOB 2L/32H/2Heads (92,690 params), trained on E5 time-based 60-second export at H10. IC=0.380, DA=64.0%, R²=0.124 on test split. Best model from 5-run ablation (no CVML, Huber loss). Holding: 10 events × 60s = **10 minutes**.

Data: `e5_timebased_60s` test split — 8,337 sequences, 35 days.

Cost model: IBKR validated ($0.70/contract commission). Deep ITM: half-spread=$0.005, delta=0.95, breakeven=1.4 bps. ATM: half-spread=$0.015, delta=0.50, breakeven=4.9 bps.

Spread filter: max_spread_bps=1.05 (1-tick only, 70.3% of samples).

### Deep ITM Options (delta=0.95)

| Threshold | Trades | 0DTE Return | Win Rate | ProfitFactor | Avg P&L/Trade |
|-----------|--------|-------------|----------|--------------|---------------|
| **0.7 bps** | **740** | **-1.93%** | **40.1%** | 0.622 | -$2.61 |
| 2.0 bps | 730 | -3.85% | 38.0% | 0.522 | -$5.27 |
| 3.0 bps | 714 | -5.73% | 37.5% | 0.499 | -$8.02 |
| 5.0 bps | 684 | -5.59% | 38.0% | 0.500 | -$8.17 |
| 8.0 bps | 594 | -1.37% | 37.0% | 0.635 | -$2.30 |
| 10.0 bps | 511 | -5.10% | 36.0% | 0.463 | -$9.99 |

### ATM Options (delta=0.50)

| Threshold | Trades | 0DTE Return | Win Rate | Avg P&L/Trade |
|-----------|--------|-------------|----------|---------------|
| 0.7 bps | 740 | -3.07% | 40.1% | -$4.15 |
| 2.0 bps | 730 | -4.07% | 38.0% | -$5.57 |
| 3.0 bps | 714 | -5.02% | 37.5% | -$7.03 |
| 5.0 bps | 684 | -4.87% | 38.0% | -$7.12 |
| 8.0 bps | 594 | -2.43% | 37.0% | -$4.09 |
| 10.0 bps | 511 | -4.14% | 36.0% | -$8.10 |

### Key Finding

E5 improved IC by **180%** (0.380 vs E4's 0.136) and test DA by **+9.6pp** (64.0% vs 54.4%), but backtest win rate **decreased** by 4.9pp (40.1% vs 45.0%). The 60s time-based bins produce much stronger signal (IC), but the 10-minute hold time exposes positions to more adverse price movement than E4's ~1 minute hold at H60.

**Best result**: Deep ITM at 0.7 bps → -1.93% (improvement from E4's -3.68%, +1.75pp). Still negative.

**Root cause persists**: DA=64% on smoothed-average labels → 40% execution win rate. The smoothed-average label (average of next 10 returns) does not equal the point-to-point return (price at t+10 vs t). The label-execution mismatch is the fundamental bottleneck across all 7 backtest rounds (R1-R7).

**Model conservatism**: Prediction std=7.35 bps vs actual return std=27.4 bps — model predicts 3.7x smaller magnitudes. At 0.7 bps threshold, 89% of predictions qualify (740/8337), confirming the model is extremely conservative.

### Comparison with Prior Rounds

| Round | Model | Sampling | Hold | IC | DA | Best Return | Win Rate |
|-------|-------|----------|------|-----|------|-------------|----------|
| R3 (H60) | HMHP class | Event-based | 6 min | — | 88.6% | -3.7% | 43.7% |
| R5 | HMHP+Ridge | Event-based | 6 min | — | — | -2.67% | 42.8% |
| R6 (E4) | TLOB 5s H60 | Time-based 5s | ~1 min | 0.136 | 54.4% | -3.68% | 45.0% |
| **R7 (E5)** | **TLOB 60s H10** | **Time-based 60s** | **10 min** | **0.380** | **64.0%** | **-1.93%** | **40.1%** |

**Signal improved massively; execution gap persists.** The next step requires addressing the label-execution mismatch directly — either via point-return training (requires non-zero IC, tested in E2/E3 with zero result), cost-embedded labels, or direct execution simulation.

---

## Round 8: E6 Calibrated Conviction — Deep ITM (2026-03-19)

Model: E5 TLOB 60s Huber (IC=0.380, DA=64.0%, 92K params, best epoch 4)
Calibration: Variance-matching (scale factor=3.73, pred_std 7.35→27.41 bps)
Data: E5 60s test split, 8,337 sequences, 35 days
Hold: 10 events × 60s = **10 minutes**
Cost: Deep ITM (delta=0.95, half_spread=$0.005, commission=$0.70, breakeven=1.4 bps)

### Deep ITM P&L (Calibrated Predictions)

| Threshold | Trades | Win Rate | Option Return |
|-----------|--------|----------|---------------|
| 1.4 bps | 742 | 48.0% | -2.87% |
| **2.0 bps** | **741** | **50.6%** | **-0.85%** |
| 3.0 bps | 740 | 45.7% | -5.06% |
| 5.0 bps | 736 | 48.2% | -3.40% |
| 8.0 bps | 724 | 47.9% | -5.95% |
| 10.0 bps | 717 | 47.7% | -6.85% |
| 15.0 bps | 698 | 47.7% | -3.28% |
| 20.0 bps | 670 | 45.5% | -5.99% |

### Comparison: R7 (Raw) vs R8 (Calibrated)

| Metric | R7 (Raw) | R8 (Calibrated) | Change |
|--------|----------|-----------------|--------|
| Best return | -1.93% (0.7 bps) | -0.85% (2.0 bps) | **+1.08pp** |
| Best win rate | 40.1% | 50.6% | **+10.5pp** |
| Prediction std | 7.35 bps | 27.41 bps | ×3.73 |
| IC | 0.380 | 0.380 | Unchanged |

### Key Finding

Calibration improved win rate by +10.5pp (40.1% → 50.6%) and best return by +1.08pp (-1.93% → -0.85%). However, **higher thresholds DECREASE win rate** — the model lacks magnitude ranking ability. Filtering on |prediction| > 20 bps produces WORSE results (45.5% win rate) than 2 bps (50.6%). This proves the model's magnitude predictions are uninformative — only the DIRECTION is predictive.

The label-level threshold analysis (E5 report §7.1: 90.8% win rate at |label|>10 bps) does NOT transfer to model predictions. The model can predict direction (DA=64%) but cannot distinguish large moves from small ones.

### Updated Comparison Table

| Round | Model | Calibrated | IC | Best Return | Best Win% |
|-------|-------|------------|-----|-------------|-----------|
| R6 | TLOB 5s H60 | No | 0.136 | -3.68% | 45.0% |
| R7 | TLOB 60s H10 | No | 0.380 | -1.93% | 40.1% |
| **R8** | **TLOB 60s H10** | **Yes (×3.73)** | **0.380** | **-0.85%** | **50.6%** |
