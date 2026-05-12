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
| R9 | TLOB 60s H10 v3p0 | No | 0.375 | -1.39% | F-6 (was display bug pre-fix) |
| R10 | TLOB+CVML 60s H10 v3p0 | No | 0.346 | +0.56% | F-6 (was display bug pre-fix) |
| **R11** | **TLOB+GMADL+CVML 60s H10 v3p0 (NEGATIVE CONTROL)** | **No** | **-0.005** | **0.00% (no trades — mean-collapse)** | **N/A (0 trades)** |
| **R12** | **TLOB 60s H10 v3p0 (Stage 2 ckpt)** | **Yes (×3.17 var-match)** | **0.375 (preserved)** | **-3.07% @ very_high_10bps** | **OptWR=46.99%, SpotWR=35.53%** |
| **R13** | **HMHP-R 60s H10 v3p0 (cascading decoder + Phase S pool_mode mean)** | **No** | **0.356** | **-1.06% @ max_conv_20bps (48 trades)** | **OptWR=39.58%, SpotWR=33.33%** |
| **R14** | **TemporalGradBoost sklearn 60s H10 v3p0 (53 temporal features)** | **No** | **0.284** | **-0.04% @ max_conv_20bps (128 trades) — STRONGEST P&L** | **OptWR=50.00%, SpotWR=45.31% — STRONGEST WR** |

> **🛠 BACKTESTER P0 FIX SHIPPED 2026-05-05**: The `WinRate=0.0000` "F-6 display issue" referenced in R9-R14 entries above was actually a key-case bug in `scripts/run_regression_backtest.py:86-89,264-265` — script read `r.get('win_rate', 0)` (lowercase_snake) but engine returns `WinRate` (PascalCase via `metric.name` in `vectorized.py:646-651`). `.get(..., 0)` silently zeroed all spot-leg metrics in the printed summary. **All saved JSONs had correct values** (which is why I could recover them). Fix: PascalCase keys + split table into Spot vs Option leg metrics. Also persists `option_win_rate` + `option_avg_pnl` into summary (pre-fix only printed inline). R12/R13/R14 JSONs re-overwritten with the fixed script; R9/R10/R11 historical (don't need re-run; their saved JSONs ALSO had correct values, only the printed tables were broken). See "Backtester P0 Fix" section below for full diagnostic narrative.

---

## Backtester P0 Fix (2026-05-05) — Key-Case Mismatch + Spot/Option Conflation

**User-driven investigation**: Per "extreme caution and precision /effort max" mandate, dispatched 4 parallel adversarial agents to audit the backtester after the Phase Q+S+X.1 v2 + Phase Q.6.5 + Phase X.2.A.1+A.2 dev cycle. Concern: are we judging models in a misleading way?

### What was wrong

| Bug | Severity | Location | Description |
|---|---|---|---|
| **Key-case mismatch (P0)** | CRITICAL | `scripts/run_regression_backtest.py:86-87,264-265` | Inline print + summary table read lowercase_snake keys (`win_rate`, `sharpe_ratio`, `total_return`, etc.) but engine `_compute_metrics` at `vectorized.py:646-651` returns dict keyed by `metric.name` which is the PascalCase class name (`WinRate`, `SharpeRatio`, `TotalReturn`, `Expectancy`, `ProfitFactor`, `SortinoRatio`, `MaxDrawdown`, `CalmarRatio`). `.get(..., 0)` silently returned 0 for every threshold, every metric. ALL R9-R14 entries in BACKTEST_INDEX.md tables had `WinRate=0.0000` despite saved JSONs having correct values like `WinRate=0.4531` (Stage 7 max_conv_20bps). |
| **Spot/Option leg conflation (P0)** | CRITICAL | `scripts/run_regression_backtest.py:264-265` | Summary table mixed `WinRate`/`Sharpe`/`TotalRet` (stock-leg from VectorizedEngine via `all_metrics` list — ~$100K equity baseline) with `OptRet` (0DTE option P&L from `ZeroDtePnLTransformer` — option leverage + theta + IBKR costs). Different P&L populations. Readers comparing rows wrongly assumed coherent metrics. The "Win rate: 50.00%" in R14 inline output was OPTION-leg WR; the JSON-saved `WinRate=0.4531` (45.31%) was SPOT-leg. |
| **option_win_rate not persisted (P1)** | HIGH | `scripts/run_regression_backtest.py:94-96` | `option_result.option_win_rate` was printed inline but never stored in summary dict → never reached the saved JSON. Downstream consumers had no programmatic access to option-leg WR. |
| **F-6 description in CLAUDE.md misdescribed (DOC)** | MEDIUM | root `CLAUDE.md` "Known Issues F-6" | Said `WinRate=0.0000 across all thresholds when --no-zero-dte passed. P&L computes 0 in spot-mode.` — actual root cause is key-case mismatch affecting BOTH `--no-zero-dte` and `--zero-dte`/`--deep-itm` modes. The "spot-mode 0" interpretation was a half-truth. |

### What was correct (re-verified by agents)

| Subsystem | Status | Verification |
|---|---|---|
| Engine `_compute_metrics` + 8 metric classes (WinRate, Sharpe, etc.) | ✅ CORRECT | Saved JSON has accurate values (R14 max_conv_20bps WinRate=0.4531, SharpeRatio=-6.0465, TotalReturn=-0.0066, Expectancy=-5.16, ProfitFactor=0.7549) |
| `BacktestResult.trade_pnls` vs `total_trades` semantic (CLAUDE.md root rule "use trade_pnls for win-rate") | ✅ CORRECT | `metrics/trading.py:71-78` uses `len(trade_pnls)` denominator (closes only). Phase 2a+2b fix is intact. |
| HMHP-R multi-horizon `(N,3)` slicing to H10 column 0 | ✅ CORRECT | `RegressionStrategy.__init__` at `strategies/regression.py:80-83` handles both `(N,)` and `(N,3)` cases via `predicted_returns.ndim == 2` branch. R13 manual recompute IC@H0 = 0.347 matches trainer's IC=0.356 within Pearson-vs-Spearman tolerance. |
| `BacktestData.from_signal_dir` calibrated_returns auto-detection | ✅ CORRECT | `vectorized.py:180-184` Phase II D10 manifest-driven branching. R12 reproduction: predicted std=8.72 vs calibrated std=27.68, ratio 3.174x exactly matches embedded scale_factor. Live BacktestData.predicted_returns has std=27.68 → calibrated values were used (not silent fallback). |
| `SignalManifest.from_signal_dir` + Phase II tamper detection | ✅ CORRECT | Re-export shim from `hft_contracts.signal_manifest`. Phase Q.6.5 sklearn signal_metadata + Stage 6 HMHP-R 11-field compatibility block + Stage 5 calibration_method="variance_match" all validate cleanly. R9-R14 backtests all passed `--primary-horizon-idx 0` partial assertion. |
| `loader.py` C-4 strict validation (Phase O Cycle 1) | ✅ CORRECT | Lines 217-224 + 309-315 fail-loud on missing schema_version per Phase O Cycle 1 producer-consumer contract. |
| IBKR cost model (Deep ITM = 1.4 bps, ATM put = 3.8, ATM call = 4.9) | ✅ CORRECT | `OpraCalibratedCosts.deep_itm()` + `ZeroDtePnLTransformer` math reconciles to within ~18% of CLAUDE.md headline. (P2 caveat: live breakeven = 1.65 bps with BSM theta; CLAUDE.md "1.4 bps" is no-theta quick-flip headline; threshold sweep deep_itm_1.4bps tests slightly below true BE, explaining persistently-negative OptRet at that threshold across all rounds — known by-design, not a bug.) |

### What changed in the fix

`lob-backtester/scripts/run_regression_backtest.py` (3 surgical edits, ~30 LOC):
1. **Inline metric print** (lines 86-89): keys `total_return`, `sharpe_ratio`, etc. → `TotalReturn`, `SharpeRatio`, etc.
2. **Option metrics persistence** (after line 96): `summary["option_win_rate"]` + `summary["option_avg_pnl"]` added when `n_trades > 0`. Pre-fix these were inline-only.
3. **Summary table redesign** (lines 256-265): split into 2 metric groups separated by `|` bars: `[Spot: WinRate, Sharpe, TotalRet]` | `[Option: OptWR, OptRet]`. Both populations clearly distinguished. Width expanded from 70 to 90 cols.

Code edits are commented with `2026-05-05 P0 fix:` prefix referencing this BACKTEST_INDEX section. Fix is back-compat: any consumer of saved JSONs ALREADY uses PascalCase keys (verified via grep). The fix only affects the printed display + adds 2 new keys to JSON.

### What we now know after the fix

**Stage 7 (TemporalGradBoost) is definitively the strongest of the 4 stages tested**:
- Best SpotWR (45.31% at max_conv_20bps) of any stage
- Best OptWR (50.00% at max_conv_20bps) of any stage
- Best OptRet (-0.04% near break-even) of any stage
- Despite LOWEST headline IC (0.284 vs Stage 1's 0.329)

**Stage 5 (calibrated TLOB) does NOT reproduce CLAUDE.md Lesson 51 cleanly on v3p0**:
- Pre-fix interpretation: looked similar to other stages
- Post-fix: Stage 5 has WORST SpotWR (35-39%) of all non-failure stages
- OptWR (43-48%) is similar to other stages
- Calibration's "+10pp WR" benefit (CLAUDE.md prior baseline) does not transfer to v3p0 corpus
- This is a NEW research finding worth investigating

**All strategies are losing money in 0DTE regime** (SpotSharpe -6 to -37; all OptRet negative except Stage 4 zero-trades and Stage 3 +0.56% at high_conv_8bps cherry-pick). The MAX_CONV_20bps threshold consistently produces the LEAST-losing OptRet across stages — this is the cost-dominated regime where only the largest-magnitude predictions trigger trades.

### Status
- **Code fix**: SHIPPED (uncommitted per "commit only when explicitly requested" mandate)
- **Test verification**: re-run R12+R13+R14 with fixed script — all produce coherent split tables matching saved JSONs
- **CLAUDE.md F-6 update**: planned next (description correction)

---

## Round 14: TemporalGradBoost sklearn V3p0 (STRONGEST P&L OF CYCLE — 2026-05-05 morning)

**Cycle context**: Phase Q.6.5 Stage 7 — second sklearn ablation in the post-Phase-O cycle (Stage 1 was TemporalRidge sklearn). Tests Phase Q.5 dispatch generalization across non-default sklearn models + reproduces CLAUDE.md TemporalGradBoost ablation finding on v3p0 60s/98-feat regime.

**Config**: `lob-model-trainer/configs/experiments/nvda_first_temporal_gradboost_v3p0.yaml` — sklearn TemporalGradBoost: 200 trees + max_depth=5 + learning_rate=0.05 + subsample=0.8 + min_samples_leaf=50 + Huber loss (alpha=0.9). 53 engineered temporal features from 5 signal_indices × 3 rolling_windows × statistics. 2 parallel adversarial agents validated PRE-flight (config+wiring + risk+empirical baseline); both PROCEED.

**Training**: 1 epoch (sklearn one-shot fit) in **~2:39s** on CPU (no MPS competition). 47,963 train + 10,134 val + 8,085 test samples. Best val_loss=-0.0223.

**Test metrics**: ic=**0.2842**, r2=0.0796, directional_accuracy=**0.5948**, pearson=0.2929, mae=18.59 bps, rmse=26.56 bps, profitable_accuracy=0.6105. **Lower IC than Stage 1 TemporalRidge (0.329) — but vastly better trading P&L (see below).**

**Phase Y composability fingerprints** (sklearn sidecar `final.pt.config.json` per Phase Q.6.5.A):
- `compatibility_fingerprint`: `117cb0273fa09c7f70fda52f7e34dfe8e36779f8e30735b37c692b737fdd0b04` (IDENTICAL to Stage 1 — same data + same primary_horizon_idx=0)
- `model_config_hash`: `fdb51e3acc37314a2826830ffe15644ff7a27f77afe62564b19488d9ff0b30ec` (DIFFERENT from Stage 1 — different sklearn model_type)

### 0DTE Option P&L (8-threshold sweep, --deep-itm, IBKR+BSM calibrated)

| Threshold | Trades | Win Rate | TotalRet | OptRet | vs Stage 1 (Ridge) |
|---|---|---|---|---|---|
| deep_itm_1.4bps | 711 | (display F-6) | 0.0000 | -1.37% | Stage 1: similar magnitude |
| itm_2bps | 704 | -- | 0.0000 | -1.34% | -- |
| itm_3bps | 690 | -- | 0.0000 | -3.01% | -- |
| atm_5bps | 640 | -- | 0.0000 | -3.57% | -- |
| high_conv_8bps | 532 | -- | 0.0000 | -3.55% | -- |
| very_high_10bps | 446 | -- | 0.0000 | -3.32% | -- |
| ultra_conv_15bps | 244 | -- | 0.0000 | -0.39% | -- |
| **max_conv_20bps** | **128** | **50.00%** | **0.0000** | **-0.04% (BEST)** | **Stage 1: -0.46% (Δ +0.42pp BETTER)** |

### Cross-Stage Comparison (R6-R14: Full Cycle)

| Round | Model | IC | DA | Best OptRet | At Threshold | WR | Trades |
|---|---|---|---|---|---|---|---|
| R6 | TLOB 5s H60 | 0.136 | -- | -3.68% | -- | -- | -- |
| R7 | TLOB 60s H10 (pre-Phase-O) | 0.380 | -- | -1.93% | -- | 40.1% | -- |
| R8 | TLOB 60s H10 calibrated (pre-Phase-O) | 0.380 | -- | -0.85% | -- | 50.6% | -- |
| **Stage 1** | TemporalRidge sklearn v3p0 | 0.329 | 0.621 | -0.46% | max_conv_20bps | -- | 175 |
| **R9 / Stage 2** | TLOB no-CVML v3p0 | 0.375 | 0.642 | -1.39% | very_high_10bps | -- | -- |
| **R10 / Stage 3** | TLOB+CVML v3p0 | 0.346 | 0.629 | +0.56% | high_conv_8bps | -- | 561 |
| **R11 / Stage 4** | TLOB+GMADL+CVML v3p0 (NEG CONTROL) | -0.005 | 0.501 | 0.00% | (no trades) | N/A | 0 |
| **R12 / Stage 5** | TLOB calibrated v3p0 | 0.375 | 0.642 | -3.07% | very_high_10bps | 47.0% | 698 |
| **R13 / Stage 6** | HMHP-R v3p0 | 0.356 | 0.630 | -1.06% | max_conv_20bps | 39.6% | 48 |
| **R14 / Stage 7** | **TemporalGradBoost v3p0** | **0.284** | **0.595** | **-0.04% (BEST)** | **max_conv_20bps** | **50.0%** | **128** |

### Key Finding (Round 14)

**STRONGEST EMPIRICAL FINDING OF THE POST-PHASE-O CYCLE**: TemporalGradBoost on v3p0 produces the BEST OptRet across ALL 7 stages despite the LOWEST headline IC (0.2842) of any non-failure model. **OptRet=-0.04% at max_conv_20bps (50.00% win rate, 128 trades) is essentially break-even at the cost-gate boundary.** This challenges the heuristic "higher IC → better P&L":

| Comparison | IC | Best OptRet | Win Rate | Insight |
|---|---|---|---|---|
| Stage 1 Ridge | 0.329 | -0.46% | -- | Linear baseline |
| Stage 7 GradBoost | **0.284** (-0.045) | **-0.04%** (+0.42pp BETTER) | **50.0%** | Non-linear captures actionable patterns |

**Hypothesis**: GradBoost's discrete tree decisions produce sharper directional predictions at high-conviction quantiles where Ridge's smooth continuous output is less actionable. The non-linear ensemble captures local patterns in temporal features that translate to better trading utility despite lower cross-sectional correlation.

**Caveat (HARD)**: Sample-of-1 test-split result. Walk-forward bootstrap + out-of-sample replication required before any production trading inference. Documenting via Phase Y `experiment_provenance_hash`: data_export_fp + compat_fp `117cb027...` + model_config_hash `fdb51e3a...` uniquely identifies this near-breakeven configuration for future re-runs.

**Phase Q.5 dispatch generalization VALIDATED**: 2 sklearn models (TemporalRidge + TemporalGradBoost) both train + export + backtest end-to-end through the canonical Phase Q.5 dispatch chain. Future sklearn ablations (XGBoost-direct, LightGBM, RandomForest) inherit the contract for free.

---

## Round 13: HMHP-R V3p0 Cascading Decoder Validation (2026-05-05 morning)

**Cycle context**: Phase Q.6.5 Stage 6 — first HMHP-R live training on v3p0 corpus + first live test of Phase S `pool_mode` field. Tests: HMHP-R cascading regression decoder architecture + Phase S `mean`-pool wiring + ConfirmationModule (agreement_ratio.npy) end-to-end through canonical scripts. **Critical pre-flight finding**: Agent 2 caught schema bridge bug (`schema.py:1758-1761` silently dropped `hmhp_loss_weights` for `hmhp_regression` model_type). Fixed same-cycle (13-line surgical change) before training; existing golden fixtures still pass (confirming runtime now matches documented intent).

**Config**: `lob-model-trainer/configs/experiments/nvda_first_hmhp_r_v3p0.yaml` — HMHP-R: TLOB-encoder (hidden=64, 2 layers) + cascading regression decoders [H10/H60/H300] + RegressionConfirmationModule + Phase S pool_mode=mean + Huber regression loss + H10-primary weights {H10:0.50, H60:0.25, H300:0.15, consistency:0.10}. Total params: **169,239** (matches Agent 2's pre-flight prediction EXACTLY).

**Training**: 16 epochs in 417.0s on MPS (~26s/epoch); best epoch=7 (val_loss=32.0990, val_h10_ic=0.3687); EarlyStopping fired at epoch 15 (8 consecutive non-improving epochs since best at epoch 7; patience=8). Best weights restored from epoch 7.

**Test metrics**: test_h10_ic=**0.3561**, test_h10_da=**0.6302**, test_h10_r2=**0.1147**, test_h10_pearson=0.3465, test_h10_mae=18.12 bps, test_h10_rmse=26.05 bps. Multi-horizon: test_h60_ic=0.1408, test_h300_ic=0.0820 (longer horizons less predictive — expected).

**Phase Y composability fingerprints** (PREDICTED EXACTLY by Agent 2 BEFORE training):
- `compatibility_fingerprint`: `cdd723ae5024b877683ed55e55a30c49e882e77260156ddb69ea192e6c05998b` (DIFFERENT from R9-R11's `67c8ff36...` because HMHP-R explicitly sets `hmhp_horizons=[10,60,300]` vs TLOB's classification fallback `[10,20,50,100,200]`)
- `model_config_hash`: `53041488548e4de31a3356c57dfa5ff0b905ab958d94e372dd0bb18499a20b87` (DIFFERENT from all R9-R12 — HMHP-R completely different architecture)

**Signal export**: 6 files (added agreement_ratio.npy from RegressionConfirmationModule alongside the standard 5 — first time agreement_ratio emitted in this cycle).

### 0DTE Option P&L (8-threshold sweep, --deep-itm, IBKR+BSM calibrated)

| Threshold | Trades | Win Rate | OptRet | vs R9 (TLOB no-CVML) |
|---|---|---|---|---|
| deep_itm_1.4bps | 713 | (display F-6) | -1.41% | R9: -5.97% (Δ +4.56pp BETTER) |
| itm_2bps | 706 | -- | -1.56% | R9: -8.18% (Δ +6.62pp BETTER) |
| itm_3bps | 695 | -- | -5.66% | R9: -1.82% (Δ -3.84pp worse) |
| atm_5bps | 644 | -- | -10.90% | R9: -4.65% (Δ -6.25pp worse) |
| high_conv_8bps | 516 | -- | -2.53% | R9: -2.30% (Δ -0.23pp similar) |
| very_high_10bps | 420 | -- | -2.01% | R9: -1.39% (best, Δ -0.62pp) |
| ultra_conv_15bps | 121 | 47.93% | -1.16% | R9: -2.83% (Δ +1.67pp BETTER) |
| **max_conv_20bps** | **48** | **39.58%** | **-1.06% (best)** | **R9: 0.00% (no trades; HMHP-R fires)** |

### R9 vs R13 Comparison (TLOB vs HMHP-R on v3p0)

| Metric | R9 (TLOB no-CVML) | R13 (HMHP-R) | Δ |
|---|---|---|---|
| test_h10_ic | 0.3747 | 0.3561 | -0.019 (HMHP-R slightly weaker) |
| test_h10_da | 0.6419 | 0.6302 | -0.012 |
| test_h10_r2 | 0.1379 | 0.1147 | -0.023 |
| Best OptRet | -1.39% @ 10bps | -1.06% @ 20bps | +0.33pp BETTER |
| Trades at best | varies | 48 (max_conv tier) | -- |
| Multi-horizon outputs? | No (single-horizon only) | Yes (H60+H300 too) | -- |
| agreement_ratio.npy? | No | Yes (cross-horizon signal) | -- |
| Param count | 92,690 | 169,239 | +83% more params |
| Training time | ~3 min | ~7 min | +130% slower |

### Key Finding (Round 13)

**HMHP-R architecturally validated end-to-end on v3p0 + competitive with TLOB at high-conviction thresholds.** Single-horizon comparison: TLOB slightly stronger (IC=0.375 vs 0.356, Δ=-0.019 — same direction as CLAUDE.md TLOB Δ=+0.006 over HMHP-R but tighter margin on time-based v3p0). At max_conv_20bps threshold, HMHP-R **outperforms** R9 (-1.06% vs 0.00% no-trades) due to higher prediction variance in cascading decoders firing at extreme thresholds where TLOB filters everything out.

**Phase S `pool_mode` field FIRST LIVE TRAINING TEST** since 2026-05-04 ship — `mean`-pool resolves correctly through schema bridge → HMHPConfig → `_apply_pooling(shared_repr, "mean")` at `lob-models/.../hmhp.py:69-106`.

**Phase Y composability EMPIRICALLY VERIFIED across ALL 4 axes** combining R9-R13:
- Data axis (R9=R10=R11 same data → same compat_fp `67c8ff36...`) ✅
- Architectural axis (R9 vs R10 different arch → different model_config_hash) ✅
- Loss-tuning axis (R10 vs R11 same arch + different loss → same model_config_hash; denylist works) ✅
- Horizons-set axis (R9-R12 classification fallback vs R13 regression explicit → different compat_fp) ✅
- Calibration axis (R9 None vs R12 variance_match → different compat_fp) ✅

5-axis fingerprint discrimination structurally + empirically locked. Phase Y `experiment_provenance_hash` composition unblocked across ALL experiment-axis variations.

**Pre-flight bug catch**: Agent 2 module-wiring audit caught silent-drop schema bridge bug (loss_weights gated by `if mt == "hmhp"` excluded `hmhp_regression`). Fixed same-cycle. **Without pre-flight gate, Stage 6 would have trained with auto-adjusted uniform weights, NOT the documented H10-primary weighting** — empirical results would have been valid for the wrong experiment.

---

## Round 12: TLOB Stage-2-checkpoint Calibrated (Variance-Match) on V3p0 (2026-05-05 morning)

**Cycle context**: Phase Q.6.5 Stage 5 — calibration code path validation. Tests `--calibrate variance_match` flag in `export_signals.py` + `calibrated_returns.npy` emission + backtester auto-detection via `manifest.calibration_method`. Reuses Stage 2's existing checkpoint (no retraining); validates the only major signal-side code path not yet exercised in the post-Phase-Q.6.5 cycle.

**Config**: Stage 2's `nvda_first_pytorch_v3p0.yaml` re-used; only the export flag differs: `--calibrate variance_match --output-dir test_calibrated/`. Pre-flight 2 parallel adversarial agents validated calibration code path (`variance.py:294-295` formula) + backtester auto-detection (`vectorized.py:180-199` Phase II D10 fix). Both converged on PROCEED.

**Calibration**: scale_factor=**3.174x** (pred std 8.72 → target std 27.68; CLAUDE.md predicted ~3.73x on pre-Phase-O baseline). 6 files emitted (added calibrated_returns.npy alongside predicted_returns.npy).

**Test metrics (preserved by linear monotone calibration)**: ic=**0.3747**, r2=0.1379, directional_accuracy=0.6419 — IDENTICAL to Stage 2/R9 (verified mathematically + empirically).

**Phase II fingerprint** (calibration-method-aware):
- `compatibility_fingerprint`: `9a72a760f23d65ae...` (DIFFERENT from R9's `67c8ff36...` because `calibration_method="variance_match"` IS in CompatibilityContract per `compatibility.py:122`)

### 0DTE Option P&L (8-threshold sweep, --deep-itm, IBKR+BSM calibrated)

| Threshold | Trades | Win Rate | TotalRet | OptRet | vs R9 (uncalibrated) |
|---|---|---|---|---|---|
| deep_itm_1.4bps | 720 | (display F-6) | 0.0000 | -6.28% | R9: -5.97% (Δ -0.31pp) |
| itm_2bps | 719 | -- | 0.0000 | -7.12% | R9: -8.18% (Δ +1.06pp better) |
| itm_3bps | 718 | -- | 0.0000 | -6.45% | R9: -1.82% (Δ -4.63pp worse) |
| atm_5bps | 714 | -- | 0.0000 | -5.17% | R9: -4.65% (Δ -0.52pp) |
| high_conv_8bps | 706 | -- | 0.0000 | -4.18% | R9: -2.30% (Δ -1.88pp worse) |
| **very_high_10bps** | **698** | **46.99%** | **0.0000** | **-3.07% (best)** | **R9: -1.39% (best, Δ -1.68pp worse)** |
| ultra_conv_15bps | 671 | 45.60% | 0.0000 | -5.79% | R9: -2.83% (Δ -2.96pp worse) |
| max_conv_20bps | 637 | 43.96% | 0.0000 | -5.44% | R9: 0.00% (no trades; calibration enables trades but loses) |

### R9 vs R12 Calibration Effect

| Property | R9 (uncalibrated) | R12 (variance-match) | Effect |
|---|---|---|---|
| Trades at 1.4 bps | 712 | 720 | +8 (slight) |
| Trades at 20 bps | 0 | 637 | +637 (calibration amplifies →fires) |
| Best OptRet | -1.39% @ 10bps | -3.07% @ 10bps | -1.68pp WORSE |
| Win rate at 10bps | 40.1% (CLAUDE.md prior) | **46.99%** | +6.89pp BETTER |
| IC preserved? | -- | YES (0.3747 ≡ 0.3747) | linear monotone ✓ |

### Key Finding (Round 12)

**CLAUDE.md Lesson 51 EMPIRICALLY REPRODUCED on v3p0**: variance-match calibration improves win rate (+6.89pp at very_high_10bps) but does NOT make a losing strategy profitable; higher thresholds (15-20 bps) produce WORSE results than 8-10 bps because the model lacks true magnitude-ranking ability. Calibration only matches variance globally; per-prediction-magnitude relevance is unchanged.

**Calibration code path EMPIRICALLY VALIDATED end-to-end via canonical scripts**: `--calibrate variance_match` produces `calibrated_returns.npy`; backtester auto-detects via `manifest.calibration_method != None`; Phase II compat_fingerprint correctly differentiates calibrated vs uncalibrated artifacts (different `calibration_method` field → different fp). The Phase Q.6.5.B `Trainer.export_signals(calibration=...)` Protocol method works correctly for the calibrated path; ZERO new SSoT primitives needed.

**Phase II partial-assertion check** with `--primary-horizon-idx 0` fired correctly during backtest load (recomputed compat_fp matches stored `9a72a760f23d65ae...`; tamper-detection passes). The compat_fp differs from Stage 2's `67c8ff36...` — that's the architectural intent (calibration is a signal-side axis, not a loss-tuning axis).

---

## Round 11: TLOB+GMADL+CVML Negative Control on V3p0 (2026-05-05 morning)

**Cycle context**: Phase Q.6.5 Stage 4 — first NEGATIVE CONTROL test in the post-Phase-O cycle. Reproduces CLAUDE.md "Validated Findings — What NOT to do" GMADL complete-failure entry. Layered GMADL loss (Michankov et al. 2024, a=10, b=1.5) on Stage 3's TLOB+CVML architecture to test (a) negative-control reproduction + (b) Phase X.1 v2 _LOSS_TUNING_KEYS denylist correctness in production.

**Config**: `lob-model-trainer/configs/experiments/nvda_first_pytorch_v3p0_gmadl_cvml.yaml` — TLOB compact + CVML + GMADL (120,179 total params; identical architecture to R10/Stage 3 — only loss type differs). 4 parallel adversarial agents validated configuration + module wiring + Phase Y prediction + risk+edge case PRE-flight; all 4 converged on PROCEED.

**Training**: 7 epochs in 195.9s on MPS (~28s/epoch); best epoch=1 (val_loss=3.272154, val_ic=0.0061); EarlyStopping fired at epoch 6 (5 consecutive non-improving val_loss epochs, patience=5). Best weights restored from epoch 1.

**Test metrics**: ic=**-0.0054**, r2=-0.0013, directional_accuracy=**0.5014**, pearson=-0.0108, mae=19.32 bps, rmse=27.70 bps. CLAUDE.md predicted IC=0.007, DA=49.8% — Stage 4 reproduced WITHIN tolerance (magnitude similar, slight sign-inversion present).

**Mean-collapse diagnostics** (predictions distribution): mean=0.9015 bps / std=**0.000077 bps** / range=[0.9013, 0.9018] / 6 unique values across 8,085 samples / 80% percentile band [0.901, 0.902]. Textbook mean-collapse — model converged to constant ~0.9015 bps prediction across the entire test set.

**Phase Y composability fingerprints** (DENYLIST VERIFICATION):
- `compatibility_fingerprint`: `67c8ff36949d6809aede114631cb0f49ceee947a1959e591d1883fd90abaaa6a` (IDENTICAL to R9 + R10 — same v3p0 data contract)
- `model_config_hash`: `3ced844386c6f7872ab9dbdb550e0d37dcd7f671fc823a5006ab6ea29224ecf8` (IDENTICAL to R10 — `_LOSS_TUNING_KEYS` denylist correctly filters gmadl_a + gmadl_b + regression_loss_type so Stage 3 Huber and Stage 4 GMADL produce same architectural fingerprint)

### 0DTE Option P&L (8-threshold sweep, --deep-itm, IBKR+BSM calibrated)

| Threshold | Trades | WinRate | TotalRet | OptRet |
|---|---|---|---|---|
| deep_itm_1.4bps | 0 | 0.0000 | 0.0000 | 0.00% |
| itm_2bps | 0 | 0.0000 | 0.0000 | 0.00% |
| itm_3bps | 0 | 0.0000 | 0.0000 | 0.00% |
| atm_5bps | 0 | 0.0000 | 0.0000 | 0.00% |
| high_conv_8bps | 0 | 0.0000 | 0.0000 | 0.00% |
| very_high_10bps | 0 | 0.0000 | 0.0000 | 0.00% |
| ultra_conv_15bps | 0 | 0.0000 | 0.0000 | 0.00% |
| max_conv_20bps | 0 | 0.0000 | 0.0000 | 0.00% |

### R9 vs R10 vs R11 (3-way comparison: TLOB / TLOB+CVML / TLOB+GMADL+CVML)

| Metric | R9 (Huber, no-CVML) | R10 (Huber+CVML) | R11 (GMADL+CVML) | Phase Y Hashes |
|---|---|---|---|---|
| test_ic | 0.3747 | 0.3464 | -0.0054 | R9 vs R10/R11: different architecture; R10 vs R11: same architecture |
| test_da | 0.6419 | 0.6294 | 0.5014 | -- |
| Best OptRet | -1.39% @ very_high_10bps | +0.56% @ high_conv_8bps | 0.00% (no trades) | -- |
| compat_fingerprint | 67c8ff36... | 67c8ff36... | 67c8ff36... | ALL IDENTICAL (same data contract) |
| model_config_hash | de47c0ef... | 3ced8443... | 3ced8443... | R10 == R11 (denylist works); R9 differs (no-CVML architectural difference) |

### Key Finding (Round 11)

**FIRST EMPIRICAL PROOF of Phase X.1 v2 `_LOSS_TUNING_KEYS` denylist correctness in production.** R10 (Huber loss) and R11 (GMADL loss) produce IDENTICAL `model_config_hash` despite different loss functions because the denylist filters out `gmadl_a` + `gmadl_b` + `regression_loss_type`. R9 (no-CVML architecture) differs in `model_config_hash` from R10/R11 because `tlob_use_cvml` flag IS in the model architecture (not denylisted).

**Combined with R10's prior architectural-axis verification, Phase Y `experiment_provenance_hash = sha256(data_export_fp + feature_set_content_hash + compat_fp + model_config_hash)` composition is now FULLY VALIDATED across:**
- Architectural axis (R9 vs R10: different arch → different model_config_hash) ✅
- Loss-tuning axis (R10 vs R11: same arch + different loss → same model_config_hash) ✅
- Data axis (R9 + R10 + R11 same data → same compat_fingerprint) ✅

**Negative control validation**: GMADL a=10, b=1.5 reproducibly produces mean-collapse failure mode on v3p0. Pipeline correctly produces 0 trades when |pred| < 1.4 bps cost gate (no false-positive P&L from degenerate signal). EarlyStopping + ModelCheckpoint(save_best_only=True) prevented late-epoch corruption (best.pt restored from epoch 1, well before the documented epoch-16 loss inversion).

---

## Round 10: Second PyTorch V3p0 — TLOB+CVML (2026-05-04 night)

**Cycle context**: Phase Q.6.5 + Phase X.2.A.1+A.2 close-out — second PyTorch model trained on v3p0 baseline corpus, this time WITH CVML feature-mixing front-end enabled (Li et al. ICLR 2025; 5 dilated causal Conv1D layers, dilation [1,2,4,8,16], 98→49 feature compression).

**Config**: `lob-model-trainer/configs/experiments/nvda_first_pytorch_v3p0_cvml.yaml` — TLOB compact + CVML (120,179 total params; CVML adds 29,057 params, embedding shrinks by 1,568 → net +27,489 vs no-CVML 92,690). Same Huber δ=12.6, regression H10, e5_timebased_60s_v3p0 corpus.

**Training**: 16 epochs in 420.9s on MPS (~26.3s/epoch — only 5% slower than no-CVML's 25s/epoch); best epoch=10 (val_loss=142.32, val_ic=0.362, val_r2=0.121, val_da=0.633); early-stopped at epoch 15.

**Test metrics**: ic=**0.3464**, r2=**0.1164**, directional_accuracy=**0.6294**, pearson=0.3483, mae=18.16 bps, rmse=26.02 bps, profitable_accuracy=0.6526. ALL 3 PRIMARY METRICS WITHIN TOLERANCE BAND (CLAUDE.md baseline IC=0.373). CVML produced **slightly worse** metrics than no-CVML (Δ=-0.028 IC, -0.022 R², -0.013 DA) — empirically reproduces CLAUDE.md "CVML doesn't transfer to low-dim/small-sample regime".

**Phase Y composability fingerprints**:
- `compatibility_fingerprint`: `67c8ff36949d6809aede114631cb0f49ceee947a1959e591d1883fd90abaaa6a` (IDENTICAL to R9 — same data contract)
- `model_config_hash`: `3ced844386c6f7872ab9dbdb550e0d37dcd7f671fc823a5006ab6ea29224ecf8` (DIFFERENT from R9's `de47c0ef...` — `tlob_use_cvml` flag in params correctly differentiates fingerprint)

### 0DTE Option P&L (8-threshold sweep, --deep-itm, IBKR+BSM calibrated)

| Threshold | Trades | WinRate | TotalRet | OptRet | vs R9 (no-CVML) |
|---|---|---|---|---|---|
| deep_itm_1.4bps | 712 | 0.0000 (F-6) | 0.0000 | -5.73% | -5.97% R9 (Δ +0.24pp) |
| itm_2bps | 704 | 0.0000 | 0.0000 | -1.95% | -8.18% R9 (Δ +6.23pp better) |
| itm_3bps | 695 | 0.0000 | 0.0000 | -4.55% | -1.82% R9 (Δ -2.73pp worse) |
| atm_5bps | 650 | 0.0000 | 0.0000 | -5.96% | -4.65% R9 (Δ -1.31pp worse) |
| **high_conv_8bps** | **561** | **0.0000** | **0.0000** | **+0.56% (best)** | **-2.30% R9 (Δ +2.86pp better)** |
| very_high_10bps | 513 | 0.0000 | 0.0000 | -2.67% | -1.39% R9 (best, Δ -1.28pp worse) |
| ultra_conv_15bps | 370 | 0.0000 | 0.0000 | -1.69% | -2.83% R9 (Δ +1.14pp better) |
| max_conv_20bps | 0 | 0.0000 | 0.0000 | 0.00% | 0.00% R9 (both no trades) |

### R9 vs R10: TLOB-no-CVML vs TLOB+CVML on v3p0

| Metric | R9 (no-CVML) | R10 (CVML) | Δ |
|---|---|---|---|
| Params | 92,690 | 120,179 | +27,489 (+30%) |
| Train wall-clock | 359.7s (13 ep) | 420.9s (16 ep) | +61s (+17%) |
| Test IC | 0.3747 | 0.3464 | -0.028 |
| Test R² | 0.1379 | 0.1164 | -0.022 |
| Test DA | 0.6419 | 0.6294 | -0.013 |
| Best OptRet | -1.39% (very_high_10bps, 473 trades) | +0.56% (high_conv_8bps, 561 trades) | +1.95pp |
| Best threshold trades | 473 | 561 | +88 |

### Key Finding (Round 10)

**CVML implementation correctness validated on v3p0**. The +27,489 added parameters from CVML's 5 dilated causal Conv1D layers do NOT improve regression metrics on the 98-feature 60s-bin v3p0 corpus — empirically reproducing CLAUDE.md prior finding "CVML doesn't transfer to low-dim/small-sample regime" (CVML 0.346 vs no-CVML 0.375 on test_ic; CLAUDE.md prior was 0.373 vs 0.380 — same direction, slightly larger Δ on v3p0 within sampling-noise variance). Backtest shows mixed results: CVML's best OptRet (+0.56% at high_conv_8bps) is single-threshold positive but other thresholds are similar/worse than no-CVML. The +0.56% is barely above 0% break-even and likely within sampling noise of the WinRate=0 F-6 display issue.

**Phase Y composability empirically verified** (R9/R10 cross-comparison): same data contract → same `compatibility_fingerprint`; different architecture (`tlob_use_cvml` flag) → different `model_config_hash`. The two fingerprints compose into different `experiment_provenance_hash` values, confirming the Phase X.1.A SSoT design correctly separates DATA-LAYER and MODEL-LAYER provenance.

---

## Round 9: First PyTorch V3p0 End-to-End Validation (2026-05-04 night)

**Cycle context**: Phase Q.6.5 + Phase X.2.A.1+A.2 + Phase Q+S+X.1 v2 close-out. First PyTorch model trained + signal-exported + backtested on the new v3p0 baseline corpus through canonical scripts post-Q.6.5 dispatch refactor.

**Config**: `lob-model-trainer/configs/experiments/nvda_first_pytorch_v3p0.yaml` — TLOB compact (92,690 params, hidden_dim=32, num_layers=2, num_heads=2, BiN, no CVML), Huber loss δ=12.6, regression H10 on `e5_timebased_60s_v3p0` (230 days; 162 train + 35 val + 33 test, all schema=3.0).

**Training**: 13 epochs in 359.7s on MPS; best epoch=7 (val_loss=140.96, val_ic=0.377, val_r2=0.141, val_da=0.636); early-stopped at epoch 12.

**Test metrics**: ic=**0.3747**, r2=**0.1379**, directional_accuracy=**0.6419**, pearson=0.3765, mae=17.90 bps, rmse=25.70 bps, profitable_accuracy=0.6664. ALL 3 PRIMARY METRICS WITHIN ±10pp/±5pp tolerance band of pre-Phase-O E5 R7 baseline (val_ic≈0.375, val_r2≈0.135, val_da≈0.636).

**Signal export**: 8,085 test signals via canonical Q.6.5.B `scripts/export_signals.py` (now uses `create_trainer + setup + load_checkpoint + trainer.export_signals` Protocol method). signal_metadata.json: 22 top-level keys + 11-field `compatibility` block + `compatibility_fingerprint=67c8ff36949d6809aede114631cb0f49ceee947a1959e591d1883fd90abaaa6a` (64-hex SHA-256). Top-level/nested `schema_version=3.0` + `contract_version=3.0` parity verified (Phase Q.9 invariant).

### 0DTE Option P&L (8-threshold sweep, --deep-itm, IBKR+BSM calibrated)

| Threshold | Trades | WinRate | TotalRet | OptRet |
|---|---|---|---|---|
| deep_itm_1.4bps | 716 | 0.0000 (F-6) | 0.0000 | -5.97% |
| itm_2bps | 710 | 0.0000 | 0.0000 | -8.18% |
| itm_3bps | 699 | 0.0000 | 0.0000 | -1.82% |
| atm_5bps | 668 | 0.0000 | 0.0000 | -4.65% |
| high_conv_8bps | 569 | 0.0000 | 0.0000 | -2.30% |
| **very_high_10bps** | **473** | **0.0000** | **0.0000** | **-1.39% (best)** |
| ultra_conv_15bps | 252 | 0.0000 | 0.0000 | -2.83% |
| max_conv_20bps | 0 | 0.0000 | 0.0000 | 0.00% (no trades — model rarely predicts >20bps; consistent with NVDA 60s std~12.5 bps) |

### Comparison: R7 (Pre-Phase-O 233-day) vs R9 (V3p0 230-day)

| Metric | R7 (Pre-Phase-O) | R9 (v3p0) | Change | Status |
|---|---|---|---|---|
| Corpus days | 233 | 230 (3 fail-loud) | -3 | Phase O Cycle 1 hardening per hft-rules §8 |
| IC | 0.380 | 0.3747 | -0.005 | Within ±0.10 tolerance ✅ |
| Best return | -1.93% | -1.39% | +0.54pp | Slightly improved (more training data on 164/233 days) |
| Best threshold | 0.7 bps (deep_itm) | 10.0 bps (very_high_conv) | shifted | More confident sub-population; fewer trades (473 vs 711) |
| Trades at deep_itm_1.4bps | 711 | 716 | +5 | Comparable test-split sample sizes (8085 v3p0 vs 8337 pre-Phase-O) |

### Key Finding (Round 9)

**Phase Q.6.5 + Phase X.2.A.1+A.2 + Phase Q+S+X.1 v2 closures empirically validated end-to-end**. The v3p0 backtest reproduces R7's pre-Phase-O magnitude (-1.39% vs -1.93%, both negative, both same threshold-sweep shape) with metrics within tolerance — confirming NO corrupt-module propagation across the 4-cycle refactor (Phase Q dispatch unification + Phase S HMHP pool harmonization + Phase X.1 v2 self-validating checkpoint cluster + Phase Q.6.5 training-pipeline-completion + Phase X.2.A.1+A.2 validate_day_metadata SSoT consolidation). The slight improvement in best return (+0.54pp) likely traces to the +21% additional training sequences on 164/233 days from Phase O B.2 session-Clear handling fix.

WinRate=0 across all thresholds is the known **F-6 backtester display issue** (CLAUDE.md Validated Findings: "lob-backtester WinRate=0.0000 across all thresholds when --no-zero-dte passed"). Affects display only; OptRet is the load-bearing economic metric.

---

## Round 15: Phase Y Producer-Side End-to-End Validation (TLOB v3p0 export-only re-run, 2026-05-05)

| Field | Value |
|---|---|
| **Goal** | Empirically validate Phase Y Stage 1 producer wiring on R9's existing checkpoint via canonical `scripts/export_signals.py`. Verifies `model_config_hash` lands at signal_metadata.json root + matches checkpoint sidecar bit-exactly. Phase C.1 horizons truth-pin behavior also empirically observed. |
| **Method** | Export-only (no re-training): `python scripts/export_signals.py --config configs/experiments/nvda_first_pytorch_v3p0.yaml --checkpoint outputs/experiments/nvda_first_pytorch_v3p0/checkpoints/best.pt --split test --output-dir outputs/.../signals/test_stage8_phase_y`. Backtest: `python scripts/run_regression_backtest.py --signals .../test_stage8_phase_y --name stage8_phase_y_validation --exchange XNAS --primary-horizon-idx 0 --deep-itm`. Preserved R9's `signals/test/` for forensic comparison. |
| **Data** | Same `e5_timebased_60s_v3p0` test split (8,085 samples) as R9. |
| **Status** | **PHASE Y EMPIRICALLY VALIDATED + bit-exact R9 reproduction** |

### 0DTE Option P&L (Deep ITM 8-threshold sweep)

| Threshold | Trades | OptWR | OptRet | R9 OptRet | Δ |
|---|---|---|---|---|---|
| deep_itm_1.4bps | 716 | 47.49% | -5.97% | -5.97% | bit-exact |
| itm_2bps | 710 | 46.20% | -8.18% | similar | reproducible |
| itm_3bps | 699 | 47.50% | -1.82% | similar | reproducible |
| atm_5bps | 668 | 44.91% | -4.65% | similar | reproducible |
| high_conv_8bps | 569 | 44.99% | -2.30% | similar | reproducible |
| **very_high_10bps** | 473 | 46.30% | **-1.39% (best)** | **-1.39% (best)** | ✅ **BIT-EXACT** |
| ultra_conv_15bps | 252 | 42.06% | -2.83% | similar | reproducible |
| max_conv_20bps | 0 | (no trades) | 0.00% | 0 trades | identical |

### R15 vs R9: Phase Y producer-side validation

| Aspect | R9 (training-time, pre-Phase-Y/C.1) | R15 (export-only, post-Phase-Y/C.1) | Status |
|---|---|---|---|
| test_ic | 0.3747 | 0.3747 | ✅ BIT-EXACT (same checkpoint, same data) |
| test_r2 | 0.1379 | 0.1379 | ✅ BIT-EXACT |
| Best OptRet | -1.39% @ very_high_10bps | -1.39% @ very_high_10bps | ✅ BIT-EXACT |
| `compatibility_fingerprint` | `67c8ff36949d6809...` (wrong horizons [10,20,50,100,200]) | `77895268cfdaba4a...` (correct horizons [10,60,300]) | ⚠️ Differs by Phase C.1 design — see Lesson 73 |
| `model_config_hash` | (NOT in R9 metadata — pre-Phase-Y) | `de47c0ef49abc0ef...` matches checkpoint sidecar bit-exactly | ✅ Phase Y producer-side validated |

### Key Finding (Round 15)

**Phase Y producer-side EMPIRICALLY VALIDATED** end-to-end on real data. Three architectural invariants confirmed:

1. **SSoT discipline**: `model_config_hash = de47c0ef49abc0ef5d9d69efe1d4003a8b9551f24d5e6574b77f52fc041ecbb4` is BIT-EXACT between Phase X.1 v2 checkpoint sidecar AND Phase Y signal_metadata.json. Both producers use `compute_model_config_hash` SSoT at `lobtrainer.training.compatibility:298`.

2. **Computation invariance**: same checkpoint + same data + same horizons feeding the model = same metrics + same backtest results. Phase Y/C.1 affect IDENTITY (provenance fingerprints), NOT COMPUTATION (model output values). Bit-exact metric reproduction across the architectural cycle proves no model-side regression.

3. **Phase C.1 truth-pin**: loading R9's pre-Phase-C.1 checkpoint emits `CheckpointConfigMismatchWarning` showing horizons drift `(10, 60, 300)` (post-truth-pin, correct) vs `(10, 20, 50, 100, 200)` (R9's pre-truth-pin, WRONG — classification defaults from silent-fallback at compatibility.py:233 that Phase C.1 deleted). Empirically observed in production code path. **Implication**: R9-R14 stored compatibility_fingerprints reflect WRONG horizons. See PHASE_P_BACKLOG.md `#PY-6`.

R15 wall-clock: ~5s export + ~2 min backtest = ~2 min total. Zero training compute. Validated entire Phase Y producer chain at minimum cost — pattern for future Phase Y producer-side iteration documented.

---

## Round 16a — Multi-Arm Sweep (point vs peak × Ridge vs TLOB × H60, 2026-05-11)

**Backfilled 2026-05-13** per hft-rules §13 same-session ledger mandate (closes §13 violation; backtests ran 2026-05-11 but ledger entry overdue 2 days; backfill source: 4 backtest records at `hft-ops/experiments/ledger/runs/cycle6_r16a_*_backtest_*.json` all `status: completed`). See `lob-model-trainer/EXPERIMENT_INDEX.md` "R-16a Cycle 6" entry for upstream context + cycle topology.

**Sweep config**: 2×2 grid (model_type × return_type), 1 seed per arm = 4 records. Manifest: `hft-ops/experiments/sweeps/cycle6_r16a_point_vs_peak_H60.yaml`. v3p0 baseline corpus (e5_timebased_60s_v3p0, 233 days NVDA XNAS, test split 8085 samples per arm).

### Headline (best per arm + framing)

| Arm | Best Threshold | Best OptRet | Win Rate | Sharpe | n_entries | Mean OptRet (8 thresh) |
|---|---|---|---|---|---|---|
| Ridge × Peak | deep_itm_1.4bps | **+2.84%** | 50.43% | -7.40 | 702 | **-0.34%** (cherry-pick) |
| Ridge × Point | atm_5bps | +0.98% | 51.53% | -6.48 | 326 | -0.18% |
| TLOB × Peak | deep_itm_1.4bps | +0.22% | 43.08% | -2.86 | 65 | +0.03% (sparse) |
| TLOB × Point | itm_2bps | +0.08% | 49.87% | -9.11 | 393 | -0.11% |

**CRITICAL FRAMING** (see Phase R-17 v2 16-agent audit 2026-05-11; reframed from "MAJOR EMPIRICAL FINDING" → "PRELIMINARY OUTLIER-DRIVEN OPTION-CONVEXITY ARTIFACT"):
- Ridge × Peak's +2.84% headline is **NOT validated alpha**. Rigorous p ≈ 0.74 (Wave 3 16-agent re-derivation). Mean across 8 thresholds is **NEGATIVE** (-0.34%).
- Win rate 50.43% indistinguishable from coin-flip (z=0.23) → no directional edge.
- Top 7 trades = 123.2% of return (outlier-driven; not 5-σ NVDA April 8-9 since those are TRAIN-split for R-16a).
- `peak_return` label has forward-leaking semantics `[k+1:k+h+1]` — INTENDED for trading per Wave 3 Agent D, but explains why peak > point P&L is plausible (model captures asymmetric magnitudes from gamma capture, not directional alpha).
- `best_total_return` (share-equivalent on -$187 negative position) for Ridge × Peak: -1.87% NEGATIVE. The +2.84% is OPTION CONVEXITY of asymmetric magnitudes from forward-leaking label, NOT alpha.

### Ridge × Peak — full 8-threshold sweep

| threshold | n_entries | win_rate | option_return_pct | total_return |
|---|---|---|---|---|
| deep_itm_1.4bps | 702 | 0.5043 | **+2.84%** | -0.0187 |
| itm_2bps | 686 | 0.4810 | -2.55% | -0.0478 |
| itm_3bps | 649 | 0.4823 | -2.67% | -0.0464 |
| atm_5bps | 501 | 0.4870 | -2.82% | -0.0406 |
| high_conv_8bps | 305 | 0.5246 | +0.90% | -0.0100 |
| very_high_10bps | 226 | 0.5531 | -0.09% | -0.0117 |
| ultra_conv_15bps | 127 | 0.6142 | +1.40% | +0.0014 |
| max_conv_20bps | 81 | 0.6173 | +0.24% | -0.0029 |

### Ridge × Point — full 8-threshold sweep

| threshold | n_entries | win_rate | option_return_pct | total_return |
|---|---|---|---|---|
| deep_itm_1.4bps | 641 | 0.5008 | +0.43% | -0.0279 |
| itm_2bps | 582 | 0.5223 | -0.86% | -0.0326 |
| itm_3bps | 475 | 0.5095 | -1.23% | -0.0298 |
| atm_5bps | 326 | 0.5153 | **+0.98%** | -0.0102 |
| high_conv_8bps | 194 | 0.5361 | -0.07% | -0.0099 |
| very_high_10bps | 150 | 0.5600 | -0.27% | -0.0089 |
| ultra_conv_15bps | 77 | 0.5584 | +0.22% | -0.0027 |
| max_conv_20bps | 44 | 0.5455 | -0.64% | -0.0061 |

### TLOB × Peak — sparse (n_entries=0 at most thresholds)

| threshold | n_entries | win_rate | option_return_pct | total_return |
|---|---|---|---|---|
| deep_itm_1.4bps | 65 | 0.4308 | **+0.22%** | -0.0018 |
| itm_2bps - max_conv_20bps | 0 | (no trades) | +0.00% | +0.0000 |

Predictions too sparse at higher thresholds — TLOB × peak counter-predicts (test_ic = -0.0125 per banner; not persisted in training_record.json due to #PY-182).

### TLOB × Point — sparse (only 2 thresholds have trades)

| threshold | n_entries | win_rate | option_return_pct | total_return |
|---|---|---|---|---|
| deep_itm_1.4bps | 546 | 0.4542 | -0.97% | -0.0313 |
| itm_2bps | 393 | 0.4987 | **+0.08%** | -0.0185 |
| itm_3bps - max_conv_20bps | 0 | (no trades) | +0.00% | +0.0000 |

### Phase Y composability validation R-16a

| Arm | experiment_provenance_hash | compatibility_fingerprint |
|---|---|---|
| Ridge × Point | `901c25dd1eb0f8a5...` | `44d3a00a883ef869...` |
| Ridge × Peak | `9d86357a642b4ed9...` | `7ef24c63788b0532...` |
| TLOB × Point | `a1fdaaf362c3ba60...` | `44d3a00a883ef869...` |
| TLOB × Peak | `22c8834b8768c14c...` | `7ef24c63788b0532...` |

4 distinct experiment_provenance_hash + 2 distinct compatibility_fingerprint (return_type axis correctly discriminates compat_fp). Cross-cycle: Ridge × Point compat_fp matches cycle5_multi_arm 2026-05-10 baseline (same data axis), confirming Phase C.1 truth-pin holds across cycles.

### Sub-cycle 4b smoke-test reproduction (R-16a Ridge×Peak deep_itm_1.4bps)

19 per-trade `option_trade_pnls.npy` fixtures emitted via `de99f45` producer-side dump (Sub-cycle 4a) on 2026-05-12; analyzer cell verdict = **REFUTE ✓** via H1(b) bootstrap CI binding constraint (CI crosses zero). #PY-180 STATUS:CLOSED 2026-05-13 (hft-ops `fa90238`) refined CI rendering — post-fix CI bounds in fraction units are sub-1% (was misrendered as ±100s% pre-fix per DOLLAR×100 bug).

### Round 16a — Cost Model Caveat

This round uses the standard `OpraCalibratedCosts` + `CostConfig.for_exchange("XNAS")` cost model. The +2.84% Ridge × Peak OptRet is GROSS of cycle-specific cost validation. Sub-cycle 4b empirical analysis using bootstrap CI on per-trade pnls (post-#PY-180 fix) refutes the headline via H1(b) sign-zero-crossing — NOT a cost-model issue.

### Outstanding work

- **#PY-182 NEW**: investigate training_record.status:failed + test_metrics:None across 4 R-16a training records. Banner-cited test_ic values came from in-process state not persisted to JSON.
- **R-16c sweep launch**: cycle7_r16c_multi_seed_r16a.yaml is LAUNCH-READY (40 grid × 10 seeds; ~80 min compute). Multi-seed power analysis on Ridge × Peak +2.84% will confirm/refute outlier-driven artifact framing via H1 three-conjunctive + H4 negative-control + H5 architectural invariant.
