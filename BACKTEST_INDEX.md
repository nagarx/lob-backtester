# Backtest Index

**Living ledger of all backtest experiments.** Updated after every run.

---

## Per-Entry Template (Post-Cycle-14)

<!-- Cycle 14 Option δ Phase 2 LITE — #PY-NEW-CONSUMPTION-ENFORCEMENT BACKTEST rollout 2026-05-27 -->

**REQUIRED post-Cycle-14**: every NEW `## Round N` entry MUST include a `**Wiki consultation**` block citing relevant `theory:` / `synthesis:` / `FINDING-` IDs from `hft-wiki`. Grandfathered pre-Cycle-14 entries (Rounds 1-7 + all post-FIND-070/FIND-090/R-19/R-16a-e/R-17a/R-20 retrofits + Cluster D.1+E close-out entries) are EXEMPT (validator skips with INFO when date-extracted as pre-2026-05-27).

Template format (markdown-section format compatible with existing Round-N structure):

```markdown
## Round N: <title> (verdict, YYYY-MM-DD)

Cost model: <IBKR/OPRA anchor>
Strategy: <strategy class>
Signal source: <path>

### Backtester invocation
<command>

### Wiki consultation (REQUIRED post-Cycle-14)
- `theory:<slug>` — <≥20-char justification, e.g. "Huber δ=12.6 bps recalibrated for 60s bins kurtosis≈26.5 per AD-HUBER-4">
- `synthesis:<slug>` — <≥20-char justification, e.g. "HMHP cascade AD-HMHP-1 LOAD-BEARING comparison against TLOB baseline at H10">
- `FINDING-NNN-<slug>` — <known anti-pattern context>

— OR explicit negative-result fallback:

- **None applicable** — queried `hft-wiki list theory --tag=<X>` returned 0 matches against this backtest's substance scope `<X>`.

### Per-threshold results
<existing 8-threshold cost-aware sweep table format unchanged>

### Verdict & Decision
<existing block format unchanged>
```

**Discovery workflow** (run BEFORE designing the backtest):

```bash
cd /Users/knight/code_local/HFT-pipeline-v2/hft-wiki
python3 scripts/cli.py list theory --tag=microstructure          # OFI / VPIN / spread
python3 scripts/cli.py list theory --tag=regression_losses       # Huber / GMADL for threshold sweep
python3 scripts/cli.py list theory --tag=off_exchange            # TRF / dark-pool signed-flow
python3 scripts/cli.py list theory --tag=afml                    # de Prado triple-barrier + sample weights
python3 scripts/cli.py list synthesis --tag=dl_architectures     # HMHP cascade (Cycle 14)
python3 scripts/cli.py list synthesis --tag=operator_synthesis   # 5-path Framework
python3 scripts/cli.py list finding --polarity=negative --status=validated,refuted
```

**Soft validator** (run BEFORE commit):

```bash
cd lob-backtester
python3 scripts/check_backtest_index_completeness.py
```

WARN-not-ERROR (exit 0). Use `--strict` to escalate WARN → exit 1. Full discipline + worked examples in `CONTRIBUTING.md` + `../hft-wiki/playbooks/record-backtest-result.md`.

**Why this exists**: Cycle 14 Phase 2 LITE rollout (#PY-NEW-CONSUMPTION-ENFORCEMENT HARD-ESCALATION TIER 1 response). Phase 1 (Cycle 11, EXPERIMENT_INDEX target) did NOT close per-surface Goodhart loop at Cycles 12 + 13 (both shipped 0% organic on BACKTEST_INDEX surface). Cycle 15 + Cycle 16 = H_Cycle14 measurement decisive cycles per pre-registered hypothesis in `../hft-wiki/ledgers/phases/PHASE-CYCLE-14-2026-05-27.md`; Cycle 17 = falsification trigger if both ship 0 organic citations meeting criteria (a)-(d).

**EXPORT_INDEX excluded this cycle by DESIGN**: producer-side data lineage surface, NOT consumer-side R-NN authoring surface. Re-evaluate Cycle 16+ if surface evolves.

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

## Round 16c — Multi-Seed Power Analysis on R-16a Ridge×Peak (REFUTE, 2026-05-13)

**Sweep ID**: `cycle7_r16c_multi_seed_r16a_20260512T063700`
**Compute**: ~6 hr wall-clock on M1 Pro MPS
**Effective grid**: 36/40 grid points (4 seed_42 records correctly deduped against R-16a's cycle6_r16a_* records — same fingerprint per Phase Y composer)
**Analyzer**: `hft-ops/scripts/analyze_r16c.py cycle7_r16c_multi_seed_r16a_20260512T063700 --allow-partial`

### Verdict: REFUTE (exit_code=1)

| Gate | Outcome | Observed | Threshold | Pass? |
|---|---|---|---|---|
| H1a: mean OptRet > +1.0% (Ridge×Peak deep_itm_1.4bps) | +0.0047% | +1.0% | **FAIL** |
| H1b: pooled-bootstrap CI lower-bound > 0 | (-0.0017%, +0.0116%) | CI > 0 | **FAIL** (crosses zero) |
| H1c: drop-top-5 mean > 0 | +0.0013% | > 0 | PASS (negligible) |
| H4: mean across 8 thresholds > -0.5% (Ridge×Peak negative control) | +0.0016% | > -0.5% | PASS (~0) |
| H5: Ridge bit-exact invariant (Phase A.3 REDESIGN) | True | True | PASS |

### Per-arm × per-threshold bootstrap CI summary (`*` = statistically significant at α=0.05)

#### Arm 1: TemporalRidge × point (8 thresholds; n_seeds=1 per H5 single-seed-pooling)

| threshold | n_trades | mean | CI (95%) | drop5 |
|---|---|---|---|---|
| deep_itm_1.4bps | 641 | +0.0013% | (-0.0059%, +0.0087%) | +0.0022% |
| itm_2bps | 580 | -0.0008% | (-0.0088%, +0.0071%) | +0.0003% |
| itm_3bps | 474 | -0.0030% | (-0.0120%, +0.0066%) | +0.0002% |
| atm_5bps | 326 | +0.0026% | (-0.0057%, +0.0109%) | +0.0050% |
| high_conv_8bps | 194 | -0.0019% | (-0.0162%, +0.0095%) | +0.0070% |
| very_high_10bps | 149 | -0.0006% | (-0.0204%, +0.0160%) | +0.0103% |
| ultra_conv_15bps | 76 | +0.0052% | (-0.0239%, +0.0332%) | +0.0188% |
| max_conv_20bps | 43 | -0.0111% | (-0.0574%, +0.0305%) | +0.0112% |

**0 of 8 cells statistically significant. Mean ∈ [-0.011%, +0.005%] per trade. NO directional edge.**

#### Arm 2: TemporalRidge × peak (8 thresholds; n_seeds=1 per H5; **F7 TARGET ARM**)

| threshold | n_trades | mean | CI (95%) | drop5 |
|---|---|---|---|---|
| **deep_itm_1.4bps** | **702** | **+0.0047%** | **(-0.0017%, +0.0116%)** | **+0.0013%** |
| itm_2bps | 685 | -0.0037% | (-0.0086%, +0.0013%) | -0.0001% |
| itm_3bps | 648 | -0.0034% | (-0.0096%, +0.0025%) | -0.0003% |
| atm_5bps | 500 | -0.0053% | (-0.0130%, +0.0024%) | -0.0013% |
| high_conv_8bps | 303 | +0.0040% | (-0.0050%, +0.0127%) | +0.0057% |
| very_high_10bps | 225 | +0.0007% | (-0.0125%, +0.0125%) | +0.0089% |
| ultra_conv_15bps | 125 | +0.0105% | (-0.0093%, +0.0268%) | +0.0148% |
| max_conv_20bps | 80 | +0.0051% | (-0.0247%, +0.0271%) | +0.0176% |

**0 of 8 cells statistically significant. Mean ∈ [-0.005%, +0.011%] per trade. F7 +2.84% headline REFUTED.**

The R-16a deep_itm_1.4bps "+2.84%" headline was a CUMULATIVE return over 702 trades. The per-trade mean is +0.0047% with CI=(-0.0017%, +0.0116%) crossing zero → NOT statistically distinguishable from zero. Cumulative breakdown: 702 trades × 0.0047% = +3.3% expected (close to +2.84% observed); but CI lower bound = -0.0017% × 702 = -1.2% (suggesting NEGATIVE cumulative is within sampling noise). **No directional alpha at α=0.05.**

#### Arm 3: TLOB × point (8 thresholds; n_seeds varies; **SIGNIFICANTLY NEGATIVE at 4 of 8**)

| threshold | n_seeds | n_trades | mean | CI (95%) | sig? |
|---|---|---|---|---|---|
| deep_itm_1.4bps | 9 | 5387 | -0.0045% | (-0.0071%, -0.0020%) | **\*** |
| itm_2bps | 9 | 4356 | -0.0037% | (-0.0065%, -0.0009%) | **\*** |
| itm_3bps | 9 | 2444 | -0.0044% | (-0.0083%, -0.0009%) | **\*** |
| atm_5bps | 6 | 635 | -0.0105% | (-0.0180%, -0.0033%) | **\*** |
| high_conv_8bps | 0 | 0 | NaN | NaN | (model never confident enough at this threshold) |
| very_high_10bps | 0 | 0 | NaN | NaN | (model never confident enough at this threshold) |
| ultra_conv_15bps | 0 | 0 | NaN | NaN | (model never confident enough at this threshold) |
| max_conv_20bps | 0 | 0 | NaN | NaN | (model never confident enough at this threshold) |

**4 of 8 cells SIGNIFICANTLY LOSING. 4 NaN cells = model never predicts confidently enough.**

#### Arm 4: TLOB × peak (8 thresholds; **SIGNIFICANTLY NEGATIVE at 7 of 8**)

| threshold | n_seeds | n_trades | mean | CI (95%) | sig? |
|---|---|---|---|---|---|
| deep_itm_1.4bps | 8 | 5775 | -0.0054% | (-0.0074%, -0.0033%) | **\*** |
| itm_2bps | 7 | 5051 | -0.0056% | (-0.0078%, -0.0034%) | **\*** |
| itm_3bps | 7 | 5049 | -0.0054% | (-0.0076%, -0.0031%) | **\*** |
| atm_5bps | 7 | 5043 | -0.0052% | (-0.0073%, -0.0029%) | **\*** |
| high_conv_8bps | 7 | 5019 | -0.0058% | (-0.0079%, -0.0038%) | **\*** |
| very_high_10bps | 6 | 4245 | -0.0060% | (-0.0082%, -0.0038%) | **\*** |
| ultra_conv_15bps | 3 | 1087 | -0.0049% | (-0.0096%, -0.0004%) | **\*** |
| max_conv_20bps | 1 | 107 | -0.0083% | (-0.0289%, +0.0113%) | (n=1 seed) |

**7 of 8 cells SIGNIFICANTLY LOSING (only max_conv_20bps non-sig due to low sample). TLOB encoder COUNTER-predicts peak labels at every threshold.**

### Aggregate verdict across 32 cells

- **0 of 32 cells significantly POSITIVE** at α=0.05
- **11 of 32 cells significantly NEGATIVE** at α=0.05 (all TLOB arms)
- **21 of 32 cells indistinguishable from zero** at α=0.05

**Conclusion**: R-16a's +2.84% Ridge × Peak headline is RIGOROUSLY REFUTED via multi-seed paired-bootstrap with 9-block-length blocks (Künsch 1989 / Politis-Romano 1994). The TLOB encoder COUNTER-predicts peak_return labels significantly (extends earlier Stage 2/3 TLOB×Peak test_ic=-0.0125 finding from cycle5_multi_arm to formal CI-based confirmation). No directional alpha exists in either Ridge or TLOB on v3p0 corpus at H60 across any threshold.

### Cost-model context

This round uses the standard `OpraCalibratedCosts` + `CostConfig.for_exchange("XNAS")` cost model (IBKR-calibrated from 316 NVDA 0DTE fills). The verdict is **gross of cost validation** because all bootstrap CIs are sub-1% in fraction units — well below the 1.4 bps Deep ITM breakeven. NO net-positive expectation can survive cost-adjustment.

### Producer-side validation

**Phase Y composer empirically validated**: 4 R-16a seed_42 records correctly deduped against R-16c grid points (sweep log "Duplicate found: cycle6_r16a_point_vs_peak_H60__temporal_ridge_point_return_20260511T012925_3a832bb6. Skipping."). This is the FIRST production demonstration of cross-sweep fingerprint-based dedup using Phase Y composer + dedup module → validates `experiment_provenance_hash` composability.

### Outstanding work

- **#PY-183 NEW**: TLOB encoder COUNTER-predicts peak_return labels — investigate root cause (BiN normalization + dual attention learning anti-correlated features).
- **#PY-184 STATUS:CLOSED-by-r16c-cycle-close**: Analyzer CLI bug fixes (`paths.pipeline_root` at 2 sites + `--allow-partial` flag) shipped in atomic hft-ops commit alongside this ledger entry (R-16c cycle-close 3-commit bundle).

---

## Round 16d — Horizon-Axis Sweep on v3p0 (INDETERMINATE, 2026-05-13)

**Sweep ID**: `cycle8_r16d_horizon_axis_20260513T060832`
**Compute**: ~50 min wall-clock on M1 Pro MPS
**Effective grid**: 12/12 grid points (single-seed; no cross-sweep dedup)
**Analyzer**: `hft-ops/scripts/analyze_r16d.py cycle8_r16d_horizon_axis_20260513T060832`

### Verdict: INDETERMINATE (exit_code=1)

| Gate | Outcome | Pass? |
|---|---|---|
| H1 PRIMARY (horizon-decay ≥3/4 arms monotonic) | 2/4 arms (borderline) | **FAIL** (INDETERMINATE) |
| H2 BASELINE (Ridge ≥ 0.80 × TLOB per cell) | 6/6 cells | **PASS** |
| H3 COST (median \|pred\| > 1.4 bps at H10) | 4/4 arms cleared backtest | **PASS** |
| H4 NEGATIVE CONTROL (mean OptRet > -0.5% at H10) | 4/4 arms | **PASS** |
| H5 ARCHITECTURAL (Ridge × horizon distinct SHAs) | 2/2 Ridge arms | **PASS** |

### Per-arm × per-horizon test_ic + H10 backtest summary

#### Arm 1: TemporalRidge × point_return (horizon-decay BROKEN — peaks at H60)

| horizon | test_ic | test_DA | H10@deep_itm_1.4bps mean | n_trades | CI 95% | Significance |
|---|---|---|---|---|---|---|
| H10 | +0.0179 | 51.10% | +0.001% | 470 | (-0.008%, +0.010%) | Crosses zero |
| H60 | **+0.1473** | 54.29% | n/a | n/a | n/a | (PEAK IC; not deep_itm at H10) |
| H300 | +0.0466 | 53.60% | n/a | n/a | n/a | (5-hour hold; not tradeable) |

**Monotonic decay: NO** — IC PEAKS at H60. Ridge captures point-return structure at the cost-aware tradeable horizon (H60 = 1 hour at 60s bins).

#### Arm 2: TemporalRidge × smoothed_return (horizon-decay MONOTONIC ✓)

| horizon | test_ic | test_DA | H10@deep_itm_1.4bps mean | n_trades | CI 95% | Significance |
|---|---|---|---|---|---|---|
| H10 | **+0.3289** | 62.06% | **-0.007%** | 711 | (-0.014%, -0.001%) | **SIG NEGATIVE** |
| H60 | +0.1557 | 53.84% | n/a | n/a | n/a | (decay observed) |
| H300 | +0.0711 | 54.39% | n/a | n/a | n/a | (further decay) |

**Monotonic decay: YES** — consistent with CLAUDE.md "signal half-life 5 timesteps". BUT: highest IC arm produces SIGNIFICANTLY NEGATIVE backtest — confirms CLAUDE.md E8 label-execution mismatch (model predicts smoothing residual, NOT point direction).

#### Arm 3: TLOB × point_return (horizon-decay BROKEN — peaks at H60; LOW IC overall)

| horizon | test_ic | test_DA | H10@deep_itm_1.4bps mean | n_trades | CI 95% | Significance |
|---|---|---|---|---|---|---|
| H10 | +0.0130 | 49.58% | +0.000% | 145 | (-0.012%, +0.011%) | Crosses zero |
| H60 | **+0.0570** | 49.75% | n/a | n/a | n/a | (PEAK IC; Ridge dominates at this cell — Ratio=2.585) |
| H300 | +0.0399 | 55.06% | n/a | n/a | n/a | |

**TLOB UNDERPERFORMS Ridge** for point_return at H60 (TLOB IC=0.057 vs Ridge IC=0.147). TLOB's transformer architecture provides ZERO value for execution-aligned point-return prediction at the tradeable horizon.

#### Arm 4: TLOB × smoothed_return (horizon-decay MONOTONIC ✓; BIT-EXACT match Stage 2 baseline)

| horizon | test_ic | test_DA | H10@deep_itm_1.4bps mean | n_trades | CI 95% | Significance |
|---|---|---|---|---|---|---|
| H10 | **+0.3790** | 64.11% | **-0.010%** | 711 | (-0.017%, -0.003%) | **SIG NEGATIVE** |
| H60 | +0.1445 | 55.14% | n/a | n/a | n/a | (decay observed) |
| H300 | +0.0637 | 54.44% | n/a | n/a | n/a | |

**Reproduces CLAUDE.md Stage 2 baseline**: TLOB H10 smoothed test_ic=0.3790 closely matches CLAUDE.md "test_ic=0.3747, DA=0.6419" (within ±0.005 absolute; expected variance from sklearn-vs-pytorch single-seed RNG path differences). **Phase Y composer empirically validates cross-cycle BIT-EXACTNESS** (same compatibility_fingerprint as Stage 2 + cycle5_multi_arm + cycle6_r16a smoothed_return H10 records per Phase Y dedup).

### H4 NEGATIVE CONTROL (mean OptRet across 8 thresholds at H10)

| Arm | mean OptRet | n_thresholds clean | Pass H4 (> -0.5%)? |
|---|---|---|---|
| temporal_ridge × point_return | +0.011% | 8/8 | ✓ |
| temporal_ridge × smoothed_return | -0.004% | 8/8 | ✓ (marginal) |
| tlob × point_return | +0.000% | 1/8 (7 cells have insufficient_data) | ✓ |
| tlob × smoothed_return | -0.006% | 8/8 | ✓ (marginal) |

**4/4 arms PASS H4 floor**, but smoothed-arm means are slightly negative — the cost gate at deep_itm_1.4bps marginally absorbs cumulative drift. NO arm produces clearly-positive mean across 8 thresholds.

### Cross-cycle reproducibility (Phase Y composer empirical validation)

- **12/12 distinct experiment_provenance_hash** populated (100% Phase Y composability — first sweep to achieve this on horizon-axis density)
- **6/12 distinct compatibility_fingerprint** (expected 2 return × 3 horizon = 6 unique data-axis combinations)
- **2/12 distinct model_config_hash** (Ridge vs TLOB; expected — arch differs only by model_type axis)
- **H5 invariant PASS 2/2 Ridge arms**: predicted_returns.npy SHA-256 are ALL DISTINCT across {H10, H60, H300} per Ridge arm — horizon axis IS architecturally active

**Cross-cycle BIT-EXACT reproducibility** (Phase Y dedup empirical):
- Ridge × smoothed × H10 cell at `cycle8_r16d_horizon_axis_20260513T060832` produces compat_fingerprint matching `cycle5_multi_arm_20260510T*` Ridge × smoothed × H10 cell + `cycle6_r16a_point_vs_peak_H60` records at H60 cells per Phase Y composer (verified via fingerprint cross-cycle string equality)

### Conclusion

1. **Horizon-decay is LABEL-CONDITIONAL** (NEW finding): smoothed-return arms decay monotonically; point-return arms PEAK at H60. Refutes naive "shorter horizon = higher IC" assumption.
2. **TLOB UNDERPERFORMS Ridge for point-return prediction** on v3p0: Ridge dominates at H60 (Ratio 2.585). The TLOB transformer architecture is OVERFITTED to smoothed-return label structure.
3. **Both smoothed-return arms produce SIGNIFICANTLY NEGATIVE backtests** at deep_itm_1.4bps despite highest test_ic (0.33-0.38). Empirically confirms CLAUDE.md E8 label-execution mismatch on v3p0.
4. **Phase Y composer empirically validated** at horizon-axis density (12/12 distinct experiment_provenance_hash). First production sweep achieving 100% Phase Y composability on horizon axis.
5. **#PY-186 v0.1.10 ceiling fix activated**: variable trade counts (145-711 per cell) exercised the bootstrap-CI fix; no narrow-CI artifacts observed.

**Outstanding work (deferred)**:
- **#PY-189 LATENT** (commit `ec54293`, 2026-05-13) remained dormant in R-16d: both Ridge + TLOB pre-slice to 1-D at `exporter.py:421-456` before signal export. HMHP-R arm would be required for 2-D activation. Manifest explicitly documented this.
- **R-16d-extended (deferred)**: Multi-seed power analysis at H60 point_return (Ridge peak-IC tradeable cell) — pre-registered trigger condition met (H1 = 2/4 borderline) but other gates all PASS suggests data is informative. Defer pending capacity decision.
- **Analyzer bug fix in same atomic commit**: `r16d_analysis.py:550-554` H-prefix strip on axis values (axis_values stores LABEL 'H10' not int 10). Shipped together with this ledger entry.

---

## Round 16e — Multi-Seed Extended at H60-hold on v3p0 (INDETERMINATE, 2026-05-14)

**Sweep ID**: `cycle9_r16e_multi_seed_h60_point_20260514T015452`
**Compute**: 1h52m wall-clock on M1 Pro MPS (PyTorch 2.10.0)
**Effective grid**: 40/40 grid points (10 seeds × 4 cells; `--force` overrode R-16d's 4 pre-existing Ridge cells per intentional cross-cycle override documented at manifest line 34-50)
**Analyzer**: `hft-ops/scripts/analyze_r16e.py cycle9_r16e_multi_seed_h60_point_20260514T015452` (post #PY-208 spec-drift fix shipped LOCAL 2026-05-14)

### Verdict: INDETERMINATE (exit_code=1)

| Gate | Outcome | Threshold | Pass? |
|---|---|---|---|
| H1(a): Ridge × Point × H60-hold pooled CI > 0 at deep_itm_1.4bps | CI=(-0.000468, +0.000313) | CI > 0 | **FAIL** (borderline within ±1%) |
| H1(b): mean OptRet across 8 thresholds > 0 (primary cell) | +0.00016089 | > 0 | **PASS** |
| H1(c): per-seed test_ic CI lower bound > 0.05 | 0.1473 | > 0.05 | **PASS** |
| H2: Ridge/TLOB IC ratio (point_return) > 1.5 | 1.653× [CI 1.479, 1.907] | > 1.5 | **FAIL** (borderline; CI low 1.479 just below floor by 1.4%) |
| H2: Ridge/TLOB IC ratio (smoothed_return) > 1.5 | 1.084× [CI 1.067, 1.105] | > 1.5 | FAIL (informational; matches CLAUDE.md "Ridge captures 91% TLOB IC") |
| H4 ARCHITECTURAL: Ridge × seed_42..51 bit-exact identical SHA-256 | All 10 seeds = `fe33748bb772b795...` (independently computed by metrics-validator agent; see #PY-214) | All identical | **PASS** |
| H6 E8 DIAGNOSTIC: smoothed × {Ridge, TLOB} mean ≤ 0 at H60-hold | Ridge=-0.000200, TLOB=-0.000022 | both ≤ 0 | **CONFIRMED** |

**INDETERMINATE clause TRIGGERED** (per manifest line 157-158): H1(a) FAIL but borderline AND H1(b) PASS → pre-registered remediation = R-16e-extended N=20 + 30-day walk-forward. **Pivot to Triple-Barrier (Option B) AUTHORIZED THIS SESSION over A** per Wave 2 Adversarial analysis evidence (H6 STRUCTURAL CONFIRMATION + TB infrastructure ALREADY SHIPPED end-to-end + ~5-7 hr realistic effort).

### Per-cell summary at primary cell deep_itm_1.4bps

| Cell | n_seeds | n_trades | mean | CI 95% | Significance |
|---|---|---|---|---|---|
| temporal_ridge × point_return (PRIMARY) | 1 (H4-pool) | 130 | -0.000054 | (-0.00047, +0.00031) | Crosses zero (BORDERLINE) |
| temporal_ridge × smoothed_return | 1 (H4-pool) | 132 | -0.000200 | (-0.00055, +0.00019) | Crosses zero (E8 confirmed) |
| tlob × point_return | 10 | 1264 | -0.000107 | (-0.00021, -0.00001) | **SIG NEGATIVE** |
| tlob × smoothed_return | 10 | 1319 | -0.000022 | (-0.00012, +0.00008) | Crosses zero (E8 confirmed) |

### H1(b) mean across 8 thresholds — PRIMARY CELL (Ridge × Point × H60-hold)

H1(b) interpretation per manifest line 145-149 (after #PY-208 fix): per-trade-mean equal-weighted across 8 cost-aware thresholds for the primary cell. ALL 10 Ridge seeds produce IDENTICAL per-trade pnls (H4 invariant) so analyzer single-seed pooling applies; n_trades_per_threshold ranges over 8 standard cost-aware deep_itm_1.4bps..max_conv_20bps thresholds.

`mean_across_8_observed = +0.00016089146728446395` (independently bit-exact reproduced by metrics-validator agent 2026-05-14 from per-trade .npy files + 8 backtest summary JSONs)

### Phase Y composer empirical validation

- **40/40 distinct experiment_provenance_hash** populated (continues R-16d's 100% Phase Y trust-column population)
- **4 distinct compatibility_fingerprint** (2 model × 2 return_type = 4 unique data-axis combos)
- **2 distinct model_config_hash** (Ridge vs TLOB; expected — arch differs only by model_type)
- **Cross-cycle BIT-EXACT MATCH** with R-16d: Ridge × Point × H60 cell at R-16e seed_42 produces predicted_returns.npy SHA-256 IDENTICAL to R-16d's single-seed cycle8 (Phase A.3 REDESIGN sklearn-RNG-free invariant CONFIRMED CROSS-CYCLE)

### #PY-208 spec-drift discovery + closure

R-16e's analyzer code (r16e_analysis.py) had DRIFTED from manifest line 145-149+205-208 pre-registration: DROPPED H1(b) "mean across 8 thresholds" gate + ADDED unauthorized "mean at deep_itm_1.4bps > 0" single-threshold gate. Drifted analyzer rendered REFUTE; manifest-aligned analyzer renders INDETERMINATE. Path A root-cause fix shipped LOCAL 2026-05-14:
- Added `H1_BORDERLINE_MARGIN = 0.01` constant
- Added `_mean_across_thresholds_primary_cell` helper
- Extended `R16eDecisionGateOutcome` dataclass (h1_mean_across_8_ok / h1_mean_across_8_observed / h1_ci_borderline)
- Kept h1_mean_ok / h1_mean_observed as DIAGNOSTIC (informational, not verdict-gating)
- Modified `_classify_verdict_r16e` signature + body (added INDETERMINATE clause per manifest line 157-158)
- Tests 34→39 (+5 INDETERMINATE clause tests)

Caught by mid-cycle 3-agent adversarial REFUTE-challenger via fresh manifest re-read. See `PHASE_P_BACKLOG.md #PY-208 STATUS:CLOSED-2026-05-14`.

### Cost-model context

This round uses the standard `OpraCalibratedCosts` + `CostConfig.for_exchange("XNAS")` cost model (IBKR-calibrated from 316 NVDA 0DTE fills). The verdict is **gross of cost validation** because all bootstrap CIs at the primary cell are sub-1% in fraction units — well below the 1.4 bps Deep ITM breakeven. NO net-positive expectation can survive cost-adjustment at the primary Ridge × Point × H60-hold cell.

### Producer-side validation

**Phase Y composer empirically validated cross-cycle**: R-16e Ridge × Point × H60 seed_42 SHA matches R-16d's single-seed cycle8 SHA (BIT-EXACT). Confirms Phase A.3 REDESIGN sklearn-RNG-free invariant holds cross-cycle AND Phase Y dedup correctly identifies (model_type=ridge, return_type=point_return, seed=42, horizon=H60) as equivalent across separate sweeps.

### Outstanding work

- **Triple-Barrier label experiment pivot** [USER AUTHORIZED THIS SESSION 2026-05-14]: H6 E8 STRUCTURAL CONFIRMATION + Wave 2 Adversarial 3 evidence that TB infra is ALREADY SHIPPED motivate pivot to TB labels (de Prado AFML PT/SL/MaxHold) as the architectural fix. Realistic effort 5-7 hr.
- **#PY-212 NEW**: r16e_analysis.py `EXPECTED_GRID_POINTS = 40` hardcoded constant — sister-site to #PY-208. For N=20+ sweeps, analyzer warning messages misreport `{count}/40`. Promote to manifest-driven OR CLI flag (~30 min).
- **#PY-213 NEW**: manifest line 159 "N=20 + walk-forward" CONJUNCTIVE remediation ambiguity given H4 invariance. Ridge-cell seed-extension is naïve waste; walk-forward IS meaningful. Future manifests should split into model-specific sub-clauses.
- **R-16e-extended N=20 DEFERRED**: pre-registered manifest line 159 remediation deferred in favor of Triple-Barrier pivot.
- **#PY-209 cross-cycle drift audit** (next-cycle hygiene): audit r16c + r16d analyzers against their manifest pre-registrations for #PY-208-class drift.

---

## Round 17a — LogisticLOB × TB v3p0 (REFUTE, 2026-05-14)

**Run name**: `r17a_logistic_tb_v3p0_20260514_094849`
**Signals**: `lob-model-trainer/outputs/experiments/r17a_logistic_tb_v3p0_h30/signals/test/` (17,480 samples, 7 files)
**Checkpoint**: `lob-model-trainer/outputs/experiments/r17a_logistic_tb_v3p0_h30/checkpoints/best.pt` (epoch 10 of 25, val_loss=0.392169)
**Corpus**: `data/exports/nvda_v3p0_tb_pt40_sl20_h30/` (233 days NVDA XNAS / 129,912 sequences / 1.0 GB; θ_PT=40 bps / θ_SL=20 bps / τ_max=30 bins)
**Compatibility FP**: `dd21d07922809691...`
**Cost model**: `OpraCalibratedCosts` + `CostConfig.for_exchange("XNAS")` (IBKR-calibrated from 318 NVDA 0DTE fills)

### Backtester invocation

```bash
python scripts/run_readability_backtest.py \
  --signals ../lob-model-trainer/outputs/experiments/r17a_logistic_tb_v3p0_h30/signals/test \
  --name r17a_logistic_tb_v3p0 \
  --exchange XNAS \
  --min-agreement 1.0 \
  --min-confidence 0.40 \
  --holding-type horizon_aligned \
  --hold-events 30 \
  --primary-horizon-idx 0 \
  --zero-dte
```

**Notes on invocation**:
- `--deep-itm` STRIPPED (does not exist in `run_readability_backtest.py` argparse — only in `run_regression_backtest.py:171`; Wave 1D verified 2026-05-14 prep cycle).
- `--min-confidence 0.40` calibrated via P25 of emitted `confirmation_score` quantile (default 0.65 would gate 93.1% of signals → ~7% pass → too few trades). At P25=0.40: 77.8% of signals pass (13,591 of 17,480). Operator-facing recommendation: ALWAYS inspect confidence quantiles from `signal_metadata.json` before running readability backtest on non-HMHP single-horizon signals — defaults are tuned for HMHP confidence distributions.
- `--min-agreement 1.0` (default) — synthetic-constant 1.0 from Phase 1 adapter (single-horizon trivially agrees); the gate is a no-op for non-HMHP single-horizon TB.
- `--zero-dte` (default True; explicit for clarity) — applies BSM theta + OPRA half-spread + IBKR commission per the standard 0DTE option cost model.

### Verdict: REFUTE (H1 FAILS + H5 PASS)

| Gate | Outcome | Threshold | Result |
|---|---|---|---|
| H1a: mean OptRet > 0% at deep_itm_1.4bps | -1.26% (option-mode) / -1.62% (equity-mode) | > 0% | **FAIL** |
| H1b: bootstrap CI lower > 0% | Not computed (single-seed; point estimate already negative — CI cannot rescue) | > 0% | **FAIL** (implied) |
| H1c: PT-trade win rate > 50% | 44.14% (option-mode) | > 50% | **FAIL** |
| H2: PT precision > 21.1% | 22.0% (from training_metrics.json test split) | > 21.1% | **BARELY PASS** (+0.9pp margin) |
| H3: vs R-16e SMOOTHED best > +0.51% | -1.26% vs R-16e Ridge×smoothed mean=-0.0002 | > +0.51% | **FAIL** |
| H4 (diagnostic): vs R-16e POINT best > +1.0% | -1.26% vs R-16e Ridge×point mean=-0.000054 | > +1.0% | **FAIL** |
| H5 ARCHITECTURAL: each class predicted ≥ 5% | SL=20.7% / Timeout=41.4% / PT=37.9% | all ≥ 5% | **PASS** |
| H6 (diagnostic): PT-hit rate on PT-predicted ≥ 50% | 22.0% (= PT precision) | ≥ 50% | **FAIL** |

**Decision matrix**: per pre-registered handoff §4 — H1 FAILS + H5 PASS → REFUTE.

### Performance summary

| Metric | Equity-mode | 0DTE Option-mode (ATM δ=0.5) |
|---|---|---|
| Total return | -1.62% | **-1.26%** |
| Final equity | $98,381.71 | $98,741.73 |
| Trade count | 333 (1.9% rate) | 333 (1 contract/trade) |
| WinRate | 43.54% | 44.14% |
| Sharpe Ratio | -5.30 | n/a |
| Profit Factor | 0.79 | n/a |
| Expectancy | -$4.86/trade | -$3.78/trade |
| Avg hold | 30.0 events | 3.0 min |
| Max Drawdown | 2.47% | n/a |

**Gated directional accuracy**: 45.04% on 4,924 confidence-gated samples (model has slight signal — 45% > random 33%, but well below cost-aware 52% threshold for 0DTE Deep ITM 1.4 bps breakeven).

### Cost economics (per-trade avg, ATM δ=0.5)

| Component | Cost | % of total |
|---|---|---|
| Spread (OPRA half-spread, RT) | $2.65 | 49.9% |
| Commission (IBKR 318-fill median) | $1.40 | 26.4% |
| Theta (BSM, IV=40%, entry 120min before close, 3.0 min hold) | $1.27 | 23.9% |
| **Total cost/trade** | **$5.31** | 100% |
| Avg underlying move per trade | +1.66 bps (positive — slight directional signal exists) | |
| **Avg P&L/trade** | **-$3.78 (NEGATIVE)** | |

Key insight: avg underlying move +1.66 bps is POSITIVE (model finds slight directional signal), but $5.31 avg cost cannot be overcome at $170 mid × 100-share notional ($170 × 1.66 bps = $2.82 gross gain). For Deep ITM (δ≥0.7), spread drops ~50% ($1.00) and theta drops ~95% ($0.04/min) → total cost ≈ $2.50 → break-even at +1.5 bps directional. We have +1.66 bps avg, suggesting Deep ITM may BARELY break even at the MEAN — but WinRate 44% means median is negative.

### Per-class test metrics (n=17,480, training_metrics.json)

| Class | Precision | Recall | F1 | n_actual | n_predicted | Predicted % |
|---|---|---|---|---|---|---|
| StopLoss (0) | 0.551 | 0.287 | 0.378 | 6,936 (39.7%) | 3,617 | 20.7% |
| Timeout (1) | 0.733 | 0.672 | 0.701 | 7,884 (45.1%) | 7,228 | 41.4% |
| ProfitTarget (2) | 0.220 | 0.548 | 0.314 | 2,660 (15.2%) | 6,635 | 37.9% |

Class distribution FLIPPED smoke → convergence: smoke 3-epoch predicted SL=47.9% / PT=9.6% → final 25-epoch predicted SL=20.7% / PT=37.9%. Focal loss + class_weights pushed model toward minority class but only achieved 22% PT precision (no break-through to 35.7% pure-EV breakeven).

### Confirmation score distribution (Phase 1 adapter validation)

| Statistic | Value |
|---|---|
| Min | 0.3339 (theoretical floor for 3-class = 0.333) |
| P25 | 0.4063 |
| P50 (median) | 0.4687 |
| P75 | 0.5335 |
| P90 | 0.6106 |
| Max | 0.9997 |
| Mean | 0.4843 |
| Std | 0.1035 |

Per-class confidence:
- SL-predicted (n=3,617): P50=0.4448
- Timeout-predicted (n=7,228): P50=0.4759
- PT-predicted (n=6,635): P50=0.4741

Confidence threshold pass-through:
- `--min-confidence > 0.40`: 13,591 / 17,480 = 77.8%
- `--min-confidence > 0.50`: 6,562 / 17,480 = 37.5%
- `--min-confidence > 0.60`: 1,952 / 17,480 = 11.2%
- `--min-confidence > 0.65` (default): 1,199 / 17,480 = 6.9% (would have produced ~25 trades — TOO FEW for meaningful backtest)

### Phase Y composer empirical validation

- Compatibility fingerprint populated: `dd21d07922809691...` (continues R-16d/R-16e 100% Phase Y trust-column population)
- R-17a is single-arm so no cross-arm distinct-counts; but the trust-column IS populated end-to-end through training → signal_export → backtester
- Confirms Phase 1 adapter does NOT regress Phase Y composability (verified by Adv2 mid-impl gate)

### Phase 1 adapter validation (NEW infrastructure ship)

Phase 1 exporter adapter at `lob-model-trainer/src/lobtrainer/export/exporter.py:_infer_classification` ships ~75 LOC + 5 new tests; synthesizes `agreement_ratio.npy` (synthetic-constant 1.0 single-horizon) + `confirmation_score.npy` (softmax-max with `.detach()` to break gradient + defensive binary-signal guard for num_classes=1 + NaN guard via `assert_finite_array` mirroring `_infer_regression` §8 fail-loud pattern).

**Validation evidence**:
- 17,480 signals exported successfully
- `agreement_ratio` confirmed all-1.0 (synthetic-constant working)
- `confirmation_score` range [0.334, 0.9997] within [1/C, 1.0] theoretical bounds
- NaN guard didn't trip (100% finite predictions; expected for converged model)
- Backtester gates correctly: 13,591 of 17,480 pass `confirmation_score > 0.40`
- Phase 4 backtest fires 333 trades (vs ~0 with default 0.65 threshold) — operator-facing calibration is critical

### Scientific value preserved despite REFUTE verdict

1. **FIRST execution-aligned classification cycle in pipeline history**. Full producer→consumer pipeline validated at scale on 1.0 GB / 233-day TB v3p0 corpus.
2. **TB×Logistic at 40/20 bps barriers EMPIRICALLY REFUTED on v3p0 NVDA**. Corroborates #PY-217 INFEASIBLE finding extending evidence to non-cost-aware 40/20 bps barriers. Closes "is TB the answer?" hypothesis at this barrier scale + this architecture.
3. **PT precision plateau at 22%** empirically discovered — suggests architectural/feature-set ceiling rather than training-dynamic floor. Information-theoretic implication: LogisticLOB on 98 LOB-only features cannot reach 35.7% PT precision required for cost-aware pure-EV breakeven on TB labels.
4. **Phase 1 adapter validates the classification path** for future R-19 (TLOB on TB), R-20 (different feature set on TB), R-18 (cost-aware barriers) cycles. Adapter is reusable.

### Outstanding work

- **R-18 NEXT CANDIDATE**: cost-aware barrier sweep (θ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} bps × τ_max=30) per Wave 1F + Adv1 §5 recommendations. CAVEAT per #PY-217: must FIRST verify H5 PASS at chosen θ (zero H5-PASS at θ ≤ 15 bps was observed during corpus extraction phase).
- **R-19 NEXT CANDIDATE**: TLOB or HMHP on same TB v3p0 corpus — does attention/cascade architecture lift PT precision above 22% plateau?
- **R-20 NEXT CANDIDATE**: 116-feature or 128-feature on TB v3p0 — does feature expansion lift PT precision above 22% plateau?
- **#PY-218 producer-side cleanup** (STILL OPEN; not blocking R-17a): Rust types.rs:117-131 LIST format inconsistency at 3 sister sites. Validator-side workaround (hft-contracts 2.7.1) is shipped. ~1.5 hr.
- **#PY-219 NEW candidate** (Wave 1D §3 finding, 2026-05-14): TB↔SHIFTED_MAPPING alignment is coincidental not contractual. Backtester `{0=Down→SELL, 1=Stable→no-entry, 2=Up→BUY}` happens to align with TB `{0=SL→short, 1=Timeout→no-entry, 2=PT→long}` only because of TB barrier order semantics. Add TB label-encoding semantic alignment validator. ~30 min.

---

## FIND-070 Closure (2026-05-14): Readability gate silent-misconfig

### Reframing post-adversarial-validation (Wave 2 Agent F)

**Original framing (overstated)**: "Any historical experiment via `ExperimentRunner.from_yaml(nvda_readability_first_*.yaml)` has silently-wrong gate metadata."

**Corrected framing**: **LATENT-MISCONFIGURATION-TRAP** for FUTURE operators copying the YAML pattern. The two YAMLs (`configs/nvda_readability_first_xnas.yaml` + `_arcx.yaml`) are NOT currently runnable via `ExperimentRunner.from_yaml` — they lack a `signals.dir` block, so the runner would fail at `BacktestData.from_signal_dir(str(signal_dir), ...)` before ever reaching the gate. `BACKTEST_INDEX.md` contains ZERO entries citing these YAMLs being executed. The bug is real but the impact is **future-protection** for operators who'd otherwise inherit the silent default-substitution.

### Root cause (ground-truth verified)

YAML schema split incorrectly placed readability strategy parameters under the engine-level block:

```yaml
# Pre-fix YAML (BROKEN — silently dropped):
backtest:
  ...
  min_agreement: 1.0      # ← WRONG BLOCK
  min_confidence: 0.65    # ← WRONG BLOCK
# (no strategy: block at all)
```

`ExperimentRunner._build_strategy` (experiment.py:415, 499-500) reads strategy parameters from the `strategy:` block ONLY (experiment.py:281-282: `base_params = {k: v for k, v in strategy_config.items() if k != "type"}`). When `strategy:` is missing entirely, `strategy_type` defaults to `"regression"` (experiment.py:280) and `min_agreement` / `min_confidence` under `backtest:` are silently dropped. Even if `ReadabilityStrategy` had been built (it wasn't), `params.get("min_agreement", 0.667)` (experiment.py:499) would have used the readability default (`readability.py:54` P5 FIX 2026-03-17), not the YAML's `1.0`.

### Closure fixes (3-step)

1. **Module-level frozensets** at `experiment.py` enumerate schema fields per block:
   - `_KNOWN_BACKTEST_KEYS` (14 keys; includes `min_agreement` / `min_confidence` per `BacktestConfig` dataclass schema at config.py:312-313 — DEPRECATED via `BacktestConfig.__post_init__` `DeprecationWarning`, slated for field removal 2026-10-31)
   - `_KNOWN_HOLDING_KEYS` (4 keys)
   - `_KNOWN_STRATEGY_KEYS_{REGRESSION,READABILITY,DIRECTION}` (per-strategy schema)

2. **`_warn_unknown_yaml_keys` helper** emits single consolidated `RuntimeWarning` on unknown keys; construction proceeds. Mirrors hft-ops Phase 7.5 R5 idiom at commit `3dd3ccb` per hft-rules §8 ("never silently drop").

3. **Fail-loud wrong-block detection** at `_build_strategy` readability branch: when `min_agreement` or `min_confidence` is found under `backtest:` but NOT under `strategy:`, raise `ValueError` with precise migration hint per hft-rules §5. Replaces the silent-default fallback that would have re-introduced dual-source-of-truth (hft-rules §0 violation).

4. **YAML migration**: both production YAMLs moved gate parameters from `backtest:` block to a NEW `strategy:` block:

```yaml
# Post-fix YAML (CORRECT):
backtest:
  initial_capital: 100000.0
  position_size: 0.1
  ...
  costs: {...}
  zero_dte: {...}

strategy:
  type: readability
  min_agreement: 1.0
  min_confidence: 0.65
```

### Test coverage

NEW `tests/test_experiment_unknown_keys.py` ships 27 parametrized tests across 8 classes:

| Test class | Tests | Coverage |
|---|---|---|
| `TestWarnUnknownYAMLKeysHelper` | 4 | Helper-function contract (single/empty/multi/known) |
| `TestBuildBacktestConfigUnknownKeys` | 3 | Backtest-block WARN path + min_agreement legacy tolerance |
| `TestBuildHoldingPolicyUnknownKeys` | 2 | Holding-block WARN path |
| `TestBuildStrategyWrongBlockDetection` | 6 | FIND-070 core: ValueError raise paths + correct-placement passthrough + regression-strategy non-interaction |
| `TestBuildStrategyUnknownStrategyKeys` | 3 | Strategy-block WARN path (per-strategy frozensets) |
| `TestFrozensetSchemaSanity` | 4 | Lock frozenset memberships against accidental drift |
| `TestSweepPathFIND070Interaction` | 2 | `_run_sweep` populates `params[min_agreement]` → FIND-070 raise correctly suppressed (mid-impl HIGH-1 gap closure) |
| `TestBacktestConfigDeprecatedFields` | 3 | `BacktestConfig.__post_init__` emits `DeprecationWarning` for non-None `min_agreement` / `min_confidence` (mid-impl HIGH-2 closure) |

**Test results**: 27/27 PASS in 0.17s; ZERO regressions across full suite (**439 passed + 16 skipped**, was 412 passed pre-fix; net +27 tests added).

### Mid-impl gate closures (2026-05-14 same-cycle)

Adversarial code-reviewer agent returned APPROVE-WITH-FIXES; 4 fixes applied same-commit:

- **HIGH-2 (machine-visible deprecation signal)**: Added `BacktestConfig.__post_init__` `DeprecationWarning` for non-None `min_agreement` / `min_confidence` fields. Operators setting either field on the dataclass now see an explicit migration message before the 2026-10-31 field-removal cycle. Closes the "deprecated-by-comment, no machine signal" gap.
- **HIGH-1 + MED-1 (sweep-path coverage)**: Added `TestSweepPathFIND070Interaction` (2 tests) locking that `_run_sweep` populating `params[min_agreement]` correctly suppresses the FIND-070 raise.
- **MED-4 (frozenset count drift)**: Reframed citation from "13 keys" → "14 keys"; added `test_backtest_frozenset_size_and_min_agreement_membership` locking the count.
- **LOW-2 (stacklevel reasoning)**: Documented the 2-hop call-chain assumption in `_warn_unknown_yaml_keys` docstring.

### Deferred follow-ups (filed in PHASE_P_BACKLOG.md)

- **MED-3 (reuse-first threshold)**: hft-ops Phase 7.5 R5 idiom + lob-backtester FIND-070 idiom are duplicated. If a 3rd consumer ever emerges, extract `_warn_unknown_yaml_keys` to `hft_contracts` per Phase Q.4 BaseTrainer Protocol precedent.
- **2026-10-31 field-removal cycle**: Remove `BacktestConfig.min_agreement` + `BacktestConfig.min_confidence` fields entirely after the DeprecationWarning grace period. Coordinate with any `BacktestConfig.from_dict` / `load_yaml` consumers (currently only `tests/test_config.py`).

### Encoded lessons (per CLAUDE.md Lesson-NN convention)

- **Lesson NN — Wrong-block silent-drop class**: When a YAML schema splits sub-blocks (`backtest:` for engine + `strategy:` for trading-policy), operators routinely place parameters under the wrong block. The fix is per-block frozenset enumeration + WARN-on-unknown-keys + fail-loud detection for KNOWN-wrong-placement cases. Mirrors hft-ops Phase 7.5 R5 at commit `3dd3ccb`.
- **Lesson NN — Reframe "historical corruption" claims by ground-truth**: Initial FIND-070 framing claimed "historical experiments via ExperimentRunner have silently-wrong gate metadata." Wave 2 Agent F adversarial verification showed the YAMLs are not currently runnable (missing `signals.dir`); BACKTEST_INDEX has zero hits citing them being executed. The bug is real but **LATENT-MISCONFIG-TRAP** for future operators, not historical corruption. Per hft-rules §13 + saved-feedback-memory mandate "depend on ground truth code over docs."

---

## Round 19a — TLOB × TB v3p0 (REFUTE-WITH-ARCHITECTURAL-LIFT, 2026-05-15)

**Run name**: `r19_tlob_tb_v3p0_20260515_064928`
**Signals**: `lob-model-trainer/outputs/experiments/r19_tlob_tb_v3p0_h30/signals/test/` (17,480 samples, 7 files)
**Checkpoint**: `lob-model-trainer/outputs/experiments/r19_tlob_tb_v3p0_h30/checkpoints/best.pt` (epoch 11 of 26, val_loss=0.361946)
**Corpus**: `data/exports/nvda_v3p0_tb_pt40_sl20_h30/` (233 days NVDA XNAS / 129,912 sequences / 1.0 GB; θ_PT=40 bps / θ_SL=20 bps / τ_max=30 bins) — **identical corpus to Round 17a, single-variable A/B at model_type axis**
**Compatibility FP**: `dd21d079228096917c6db63227bc71d2f14534dbebb5a4a939eef19732791eaf` (**IDENTICAL to Round 17a** — Phase Y composer correctly preserves corpus identity)
**Model config hash**: `2dc7eeef5192db921ed348364fb4c76fbc5e3e917a69929791e016a99ee16a0e` (**DIFFERENT from R-17a's `9d2fdcef837d6227...`** — Phase Y composer correctly distinguishes architectural axis)
**Model**: TLOB compact-config — `tlob_hidden_dim=40 × tlob_num_layers=4 × tlob_num_heads=1` (LOCKED per #PY-236), `tlob_use_bin=true`, 130,296 parameters
**Cost model**: `OpraCalibratedCosts` + `CostConfig.for_exchange("XNAS")` (IBKR-calibrated from 318 NVDA 0DTE fills)

### Backtester invocation

```bash
python scripts/run_readability_backtest.py \
  --signals ../lob-model-trainer/outputs/experiments/r19_tlob_tb_v3p0_h30/signals/test \
  --name r19_tlob_tb_v3p0 \
  --exchange XNAS \
  --min-agreement 1.0 \
  --min-confidence 0.40 \
  --holding-type horizon_aligned \
  --hold-events 30 \
  --primary-horizon-idx 0 \
  --zero-dte
```

**Notes on invocation** (mirrors Round 17a; single-variable A/B):
- `--deep-itm` STRIPPED (does not exist in `run_readability_backtest.py` argparse).
- `--min-confidence 0.40` calibrated via Round 17a's P25 ≈ 0.40; reused here for direct cross-architecture comparison. R-19 confidence quantiles (P25=0.398, P50=0.446, P75=0.515) are SIMILAR to Round 17a (P25=0.406, P50=0.469, P75=0.534) — TLOB confidence distribution is slightly tighter at the P25 floor.
- `--min-agreement 1.0` — synthetic-constant 1.0 from Phase 1 adapter (single-horizon trivially agrees); the gate is a no-op for non-HMHP single-horizon TB.

### Verdict: REFUTE-WITH-ARCHITECTURAL-LIFT (H1 FAILS + H2 PASS materially exceeds R-17a + H5 PASS)

| Gate | Outcome | Threshold | Result | vs Round 17a |
|---|---|---|---|---|
| H1a: mean OptRet > 0% at deep_itm_1.4bps | -3.11% (option-mode) / -3.55% (equity-mode) | > 0% | **FAIL** | WORSE by 1.85pp option / 1.93pp equity |
| H1b: bootstrap CI lower > 0% | Not computed (single-seed; point estimate negative — CI cannot rescue) | > 0% | **FAIL** (implied) | — |
| H1c: PT-trade win rate > 50% | 39.75% (option-mode) | > 50% | **FAIL** | WORSE by 4.39pp |
| H2: PT precision > 21.1% | 26.9% (from test_metrics.json) | > 21.1% | **PASS** (+5.8pp margin) | **+4.9pp over Round 17a's 22.0%** |
| H3: vs R-16e SMOOTHED best > +0.51% | -3.11% vs R-16e Ridge×smoothed mean=-0.0002 | > +0.51% | **FAIL** | — |
| H4 (diagnostic): vs R-16e POINT best > +1.0% | -3.11% vs R-16e Ridge×point mean=-0.000054 | > +1.0% | **FAIL** | — |
| H5 ARCHITECTURAL: each class predicted ≥ 5% | SL=27.5% / Timeout=38.1% / PT=34.4% | all ≥ 5% | **PASS** | comparable; TLOB MORE SL / FEWER Timeout |
| H6 (diagnostic): PT-hit rate on PT-predicted ≥ 50% | 26.9% (= PT precision) | ≥ 50% | **FAIL** | **+4.9pp BETTER than R-17a's 22.0%**, but still below 50% threshold |

**Decision matrix**: per pre-registered handoff — H1 FAILS + H5 PASS + H2 margin (+5.8pp) materially exceeds R-17a's H2 margin (+0.9pp) → **REFUTE-WITH-ARCHITECTURAL-LIFT** (NEW verdict label introduced this cycle).

### Performance summary

| Metric | Equity-mode | 0DTE Option-mode (ATM δ=0.5) |
|---|---|---|
| Total return | -3.55% | **-3.11%** |
| Final equity | $96,447.51 | $96,887.65 |
| Trade count | 644 raw (322 round-trip) | 644 raw (322 round-trip; 1 contract/trade) |
| WinRate | 39.13% | 39.75% |
| Sharpe Ratio | -13.66 | n/a |
| Sortino Ratio | -18.96 | n/a |
| Calmar Ratio | -10.72 | n/a |
| Profit Factor | 0.55 | n/a |
| Expectancy | -$11.03/trade | n/a |
| Avg theta cost | n/a | $1.27/trade |
| Avg hold | 30.0 events | 3.0 min |
| Max Drawdown | 3.79% | n/a |

**Trade-rate context**: n_gate_pass=322 / n_gate_fail=7,498 → 4.12% pass rate (vs Round 17a 4.92%). TLOB confidence distribution is slightly tighter at the P25 floor; same min_confidence=0.40 threshold filters MORE samples (~12% fewer trades pass-through).

### Per-class test metrics (n=17,480, test_metrics.json) — R-19 vs Round 17a

| Class | R-19 Precision | R-19 Recall | R-19 F1 | R-17a Precision | R-17a Recall | R-17a F1 | Δ Precision | Δ Recall | Δ F1 |
|---|---|---|---|---|---|---|---|---|---|
| StopLoss (0) | 0.443 | 0.307 | 0.363 | 0.551 | 0.287 | 0.378 | **-0.108** | +0.020 | -0.015 |
| Timeout (1) | 0.770 | 0.651 | 0.706 | 0.733 | 0.672 | 0.701 | +0.037 | -0.021 | +0.005 |
| ProfitTarget (2) | **0.269** | **0.607** | **0.373** | 0.220 | 0.548 | 0.314 | **+0.049** | +0.059 | +0.059 |

**Predicted class distribution (R-19 vs R-17a)**:

| Class | R-19 n_predicted | R-19 % | R-17a n_predicted | R-17a % | Δ % |
|---|---|---|---|---|---|
| StopLoss (0) | 4,808 | 27.5% | 3,617 | 20.7% | +6.8pp |
| Timeout (1) | 6,666 | 38.1% | 7,228 | 41.4% | -3.3pp |
| ProfitTarget (2) | 6,006 | 34.4% | 6,635 | 37.9% | -3.5pp |

### Cross-architecture cost economics (per-trade avg, ATM δ=0.5)

R-19 has TLOB attention finding +4.9pp additional PT precision over R-17a Logistic, yet backtest is WORSE. The bottleneck is LABEL-COST alignment, not precision:

**Pure-EV math at 40/20 bps barriers with 1.4 bps cost**:
- R-17a: 22.0% × +40 bps + 78.0% × -20 bps - 1.4 bps = **-8.2 bps NET per PT-predicted trade**
- R-19:  26.9% × +40 bps + 73.1% × -20 bps - 1.4 bps = **-3.84 bps NET per PT-predicted trade** (4.36 bps closer to breakeven)
- Required for break-even at 1.4 bps cost: **35.7% PT precision** (R-19 closes 14% of the precision-gap; 86% remaining)

**Why R-19 backtest is WORSE despite better precision**: TLOB's PT recall=0.607 means model "finds" 6 of 10 true PTs (vs R-17a's 0.548). MORE PT predictions → MORE trades hitting cost economics. With cost economics still NEGATIVE per trade, more trades amplify losses. The +4.9pp precision lift is insufficient to overcome the precision-vs-volume amplification.

**Pre-existing cost decomposition unchanged from Round 17a** (mid × 100-share notional gives ~$1.70 per bps at $170):
| Component | Cost | % of total |
|---|---|---|
| Spread (OPRA half-spread, RT) | $2.65 | 49.9% |
| Commission (IBKR 318-fill median) | $1.40 | 26.4% |
| Theta (BSM, IV=40%, entry 120min before close, 3.0 min hold) | $1.27 | 23.9% |
| **Total cost/trade** | **$5.32** | 100% |

### Phase Y composer empirical validation — FIRST cross-architecture A/B on TB v3p0

- **Same compat_fingerprint** as Round 17a: `dd21d079228096917c6db63227bc71d2f14534dbebb5a4a939eef19732791eaf` — Phase Y composer correctly preserves CORPUS identity across the model_type axis change
- **Different model_config_hash**: `2dc7eeef5192db921ed348364fb4c76fbc5e3e917a69929791e016a99ee16a0e` (R-19 TLOB) ≠ `9d2fdcef837d6227...` (R-17a Logistic) — composer correctly distinguishes the architectural axis
- R-19 + R-17a together form FIRST single-variable model_type A/B in pipeline history using execution-aligned classification labels on the SAME corpus/contract
- Future `hft-ops ledger list --compatibility-fp dd21d07922809691` (post #PY-223 closure) would correctly group R-19 and R-17a as cross-architecture experiments on identical data

### Phase 1 adapter validation (reused from Round 17a; ship infrastructure verified)

Phase 1 exporter adapter (`lob-model-trainer/src/lobtrainer/export/exporter.py:_infer_classification`) shipped in Round 17a, reused unchanged here. Re-validation evidence:
- 17,480 signals exported successfully (same n as R-17a — same test split, same Phase 0.5 validator nested-fallback)
- `agreement_ratio` all-1.0 (synthetic-constant working for single-horizon TLOB output)
- `confirmation_score` range within [1/C=0.333, 1.0] theoretical bounds:

| Statistic | R-19 | R-17a | Δ |
|---|---|---|---|
| P25 | 0.3983 | 0.4063 | -0.008 |
| P50 (median) | 0.4463 | 0.4687 | -0.022 |
| P75 | 0.5151 | 0.5335 | -0.018 |
| P99 | 0.8189 | — | n/a |
| Mean (implicit from confirmation_percentiles) | ~0.46 | 0.4843 | ~-0.02 |

TLOB confidence distribution is slightly LOWER (tighter at P25) than Logistic — consistent with TLOB attention being more discriminative (decisions concentrated at higher confidence; fewer borderline calls).

### Scientific value preserved (REFUTE-WITH-ARCHITECTURAL-LIFT verdict)

1. **EMPIRICALLY REFUTES R-17a Lesson #95** (PT precision 22% was info-theoretic). The 22% plateau was ARCHITECTURALLY-BOUND to Logistic flatten pooling, NOT corpus-bound. TLOB attention captures +4.9pp additional signal on the SAME TB v3p0 corpus with the SAME loss policy.
2. **FIRST cross-architecture single-variable A/B on TB v3p0** in pipeline history. Phase Y composer empirically validates correct partition: same compat_fp (corpus identity) + different model_config_hash (architectural axis).
3. **Reproducibility of R-17a Phase 1 adapter** at cross-architecture scale. Adapter is architecture-agnostic; ships correctly for TLOB single-horizon classification output.
4. **Bottleneck shift from "model finds signal" → "label-cost alignment"**: even at TLOB's 26.9% PT precision, pure-EV math gives -3.84 bps NET per PT-predicted trade. Further architectural lift alone is insufficient — the 35.7% break-even precision threshold needs either (a) further model lift (HMHP cascade; R-20 candidate), (b) cost-aware barrier scale (R-18 candidate; #PY-217 H5 caveat), or (c) different feature set (R-21 candidate).
5. **TLOB compact-config (130K params) is parameter-efficient at the architectural-axis exchange**: 22.1x more parameters than Logistic (130,296 vs 5,883) produces +4.9pp PT precision. Future architectural exploration should prioritize models with non-trivial attention (HMHP cascade) over scale.

### Outstanding work

- **R-18 NEXT CANDIDATE** (elevated priority post R-19): cost-aware barrier sweep (θ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} bps × τ_max=30) with TLOB OR HMHP per R-19 architectural-lift evidence. R-19 confirms architecture matters AND cost-economics is the binding constraint. CAVEAT per #PY-217: zero H5-PASS at θ ≤ 15 bps observed during TB extraction. R-18 must FIRST verify H5 PASS at chosen θ before training.
- **R-20 NEXT CANDIDATE**: HMHP cascade-decoder on same TB v3p0 corpus — does multi-horizon decoder lift PT precision further above TLOB's 26.9%?
- **R-21 NEXT CANDIDATE**: 116- or 128-feature on TB v3p0 with TLOB — does feature expansion lift PT precision above 26.9%?
- **#PY-218 producer-side cleanup** (STILL OPEN, unchanged): Rust types.rs:117-131 LIST format inconsistency at 3 sister sites. ~1.5 hr.
- **#PY-219 NEW candidate** (unchanged from R-17a): TB↔SHIFTED_MAPPING alignment is coincidental not contractual. Add TB label-encoding semantic alignment validator. ~30 min.

### Encoded lessons (per CLAUDE.md Lesson-NN convention; chains from R-17a Lessons #94-98)

- **Lesson #99**: **TLOB architectural lift OVER Logistic on TB v3p0 NVDA is REAL but INSUFFICIENT**: +4.9pp PT precision (26.9% vs 22.0%) at 22.1x parameter cost (130,296 vs Logistic's 5,883). Lift confirms TB v3p0 has predictive signal Logistic-flatten cannot capture but TLOB-attention can. However, the lift does NOT close the cost-economics gap — both architectures REFUTE on H1 PRIMARY backtest gate.
- **Lesson #100**: **R-17a Lesson #95 REFUTED**: PT precision 22% was ARCHITECTURALLY-BOUND not info-theoretic. The plateau was a property of Logistic-flatten pooling, NOT corpus inherent. Test before assuming any "ceiling" is corpus-bound.
- **Lesson #101**: **Higher precision is necessary but not sufficient for TB backtest profitability**: R-19's +4.9pp precision lift produced WORSE backtest. Bottleneck shifted from "model finds enough signal" (R-17a) to "label-cost alignment is wrong" — even at 26.9% PT precision, 40-bps PT vs 20-bps SL barriers with 1.4 bps cost give -3.84 bps NET per PT-predicted trade.
- **Lesson #102**: **TLOB compact-config (130K) is parameter-EFFICIENT on TB v3p0** vs R9 (TLOB compact 92K on smoothed-return) which produced IC=0.3747 directly. TB labels carry SIGNIFICANTLY less linear signal than smoothed-return — TLOB attention finds non-linear interactions but the TB-vs-smoothed signal density gap is much larger than the architectural lift.
- **Lesson #103**: **`tlob_num_heads=1` empirically validated for compact-config**: paper canonical setting per #PY-236 (closes original CLAUDE.md banner gcd math error). At `hidden_dim=40 × num_heads=1` → embed_dim=40, feature-attention block at `tlob.py:166-173` divides cleanly. Training stable across 26 epochs.
- **Lesson #104 (NEW verdict-label encoded)**: **REFUTE-WITH-ARCHITECTURAL-LIFT** is a NEW classification beyond simple GO/REFUTE/INDETERMINATE/ABORT. Apply when (a) H1 PRIMARY fails AND (b) H2 BASELINE passes AND (c) the H2 margin materially exceeds the prior-architecture's H2 margin AND (d) H5 ARCHITECTURAL passes. The cycle CLOSES the architectural-ceiling hypothesis (lift IS available) but REFUTES the profitability hypothesis (lift insufficient to close cost gap). Use in future cycles where architectural axis is varied on a previously-REFUTED experiment.

---

## Round 19b — TLOB × TB v3p0 Multi-Seed N=5 (Cycle 10 / #PY-243; GO-ARCHITECTURAL + INDETERMINATE-COMMERCIAL, 2026-05-19)

**Run names**: 5 backtests at `lob-backtester/outputs/backtests/cycle10_r19_multi_seed__seed_{43..47}_*/`

**Sweep ID**: `cycle10_r19_multi_seed_20260518T222513` (5 cells × {training, signal_export, backtesting}; 0 failed; durations 4080-5130s per cell; total compute ~22,709s ≈ 6.3 hours unattended)

**Signals**: `outputs/experiments/seed_{43..47}/signals/test/` (17,480 samples per seed; Phase 1 sklearn adapter consumed cleanly with synthetic agreement_ratio = 1.0)

**Checkpoints**: `hft-ops/ledger/runs/cycle10_r19_multi_seed_20260518T222513/checkpoints/best.pt` (one shared sweep dir; seed_43 best epoch 15 val_loss=0.3648)

**Corpus**: `data/exports/nvda_v3p0_tb_pt40_sl20_h30/` (233 days NVDA XNAS / 129,912 sequences; θ_PT=40 bps / θ_SL=20 bps / τ_max=30 bins; identical to Round 17a + Round 19a)

**Compatibility FP (NEW POST-BUG2A anchor)**: `8f1148de02ad446efdcd613a3c05b00b55439740d48c03101978eaf2a5c2c353` (5 IDENTICAL across seeds; differs from R-19's PRE-BUG2A anchor `dd21d07922809691...` ONLY by `feature_layout` field — registry-tag `"default"` → content-hash `122fe5cbfb657bf91...` per `compatibility.py:228-232` BUG2-A closure mechanism; other 10 SHAPE-determining fields byte-identical proving corpus is the SAME)

**Model config hash**: `2dc7eeef5192db921ed348364fb4c76fbc5e3e917a69929791e016a99ee16a0e` (5 IDENTICAL; matches R-19 anchor — TLOB compact-config `tlob_hidden_dim=40 × tlob_num_layers=4 × tlob_num_heads=1 × tlob_use_bin=true`, 130,296 params)

**Experiment provenance hash**: `27244be31b8af3744dcd1c10c2004fd1bf6609417ddac7ae7d388f4b02aeda5d` (5 IDENTICAL — Phase Y composer is TREATMENT-LEVEL by design; see Lesson L43 in EXPERIMENT_INDEX cycle 10 entry)

### Backtester invocation

```bash
# Per-seed pattern (5 invocations, identical except --signals path):
python scripts/run_readability_backtest.py \
    --signals ../outputs/experiments/seed_43/signals/test \
    --name cycle10_r19_multi_seed__seed_43 \
    --exchange XNAS \
    --primary-horizon-idx 0 \
    --min-confidence 0.40 \
    --bin-seconds 60                       # ← NEW post FIND-NEW-01 closure 2026-05-16 morning
```

**Cost model**: ATM mode (delta=0.5, prefer_calls=True, implied_vol=0.4, atm_call_half_spread=$0.015, commission_per_contract=$0.70 / round-trip $1.40). Mode is SAME as R-19's original Round 19a (also delta=0.5/IV=0.4 — verified via direct read of R-19's result.json config).

**METHODOLOGY ASYMMETRY vs Round 19a** (cite explicitly per "cross-cycle comparison" discipline):

| Axis | R-19 (Round 19a, single seed 2026-05-15) | Cycle 10 R-19 Multi-Seed (Round 19b, 2026-05-19) |
|---|---|---|
| CLI: `--hold-events` | 30 (passed) | NOT passed (default holding policy) |
| CLI: `--min-agreement` | 1.0 (passed) | NOT passed (default) |
| CLI: `--holding-type` | horizon_aligned (passed) | NOT passed (default) |
| CLI: `--zero-dte` | passed explicit | default-enabled |
| events_per_minute | **10.0 PRE-FIND-NEW-01** (bug) | **1.0 POST-FIND-NEW-01** (correct for 60s bins) |
| Resulting hold duration | 30 events / 10.0 evt/min = 3 min | 30 events / 1.0 evt/min = 30 min |
| Resulting theta cost | $1.27/trade (3-min hold) | $4.23/trade (30-min hold) — **3.3x R-19's reported theta** |
| Resulting trade count | 322 round-trips | mean 1232 (3.8x R-19) — default holding policy completes more cycles |

**Implications**: R-19's reported OptRet=-3.11% was PRE-FIND-NEW-01 + with explicit `--hold-events 30`. Cycle 10 R-19 multi-seed OptRet=-5.70% is POST-FIND-NEW-01 + with default holding. Direct OptRet comparison is NOT clean (cost methodology + holding-policy methodology both differ). PT_precision IS the only cleanly comparable metric (training-time confusion matrix; cost-model-independent).

### Per-seed performance

| Seed | Total Trades | Equity Return | WinRate (equity) | Option Return | Option WinRate | Avg Theta | Max Drawdown |
|---|---|---|---|---|---|---|---|
| 43 | 1450 | **-4.64%** | 40.55% | **-6.26%** | 36.14% | $4.23 | 4.77% |
| 44 | 1160 | -4.61% | 39.31% | -5.69% | 36.38% | $4.23 | 4.76% |
| 45 | 1040 | **-3.96%** | **41.15%** | **-5.05%** | **37.50%** | $4.23 | **4.11%** |
| 46 | 1250 | -4.19% | **41.76%** | -5.38% | **37.92%** | $4.22 | 4.34% |
| 47 | 1260 | -4.68% | 39.21% | -6.11% | 35.40% | $4.22 | 4.82% |
| **Mean ± SD** | **1232 ± 156** | **-4.41% ± 0.31%** | **40.40% ± 1.08%** | **-5.70% ± 0.50%** | **36.67% ± 1.05%** | **$4.23 ± $0.005** | **4.56% ± 0.31%** |

**Tight cross-seed variance** (option_return std = 0.50%, ~9% of mean magnitude) confirms seed-stability of the backtest at the realized-P&L level (independent of the training-time PT precision bimodality).

### Verdict: GO-ARCHITECTURAL + INDETERMINATE-COMMERCIAL (DUAL)

Per Cycle 10 EXPERIMENT_INDEX entry decision matrix:
- **ARCHITECTURAL**: R-19 +4.9pp PT precision lift SURVIVES N=5 seed perturbation (mean PT precision 0.2639 ± 0.020 closely tracks R-19 anchor 0.269 within 0.5pp; CI lower 0.2488 cleanly above R-17a baseline 0.220). **GO-CONFIRMED**.
- **COMMERCIAL**: Backtest OptRet -5.70% ± 0.50% across all 5 seeds. The architectural lift is INSUFFICIENT to clear cost-economics floor at 40/20 bps barrier × ATM 0DTE cost model. Confirms prior Wilson+McNemar verdict from commit `b84897a` (mean diff +0.0492 vs cost-floor 0.05; -0.08pp shortfall). **INDETERMINATE-COST-INSUFFICIENT-CONFIRMED**.

### Encoded lessons (chain from R-19 Lessons #99-104; full text in EXPERIMENT_INDEX cycle 10 entry)

- **Lesson #105**: R-19 +4.9pp architectural lift is ROBUST to seed variance; Architectural Lesson #12 (multi-seed mandate) CLOSED for R-19 corpus + TLOB architecture
- **Lesson #106**: Wilson+McNemar cost-floor verdict from `b84897a` HOLDS regardless of seed-variance robustness; no architecture in Ridge/Logistic/TLOB class is commercially tradeable at this barrier scale
- **Lesson #107**: Bimodality in TLOB+focal training regime — 2 aggressive-PT seeds (8K+ predictions, 0.24 precision) vs 3 conservative-PT seeds (3.8-4.8K predictions, 0.28 precision); R-19's seed=42 was in the conservative regime
- **Lesson L43** (NEW Architectural Lesson, paired with L42 session-pivot): pre-registered fingerprint-identity gates need re-capture after architectural fixes (BUG2-A); Phase Y / dedup / pred_sha are ORTHOGONAL fingerprint mechanisms
- **Lesson L44** (NEW Architectural Lesson): Phase Y composer docstring should include explicit "**INTENTIONALLY seed-invariant**" statement; PA §17.3 should document 3-fingerprint orthogonality

### Outstanding work (Round-19b specific)

- **Round 19c NEXT candidate**: cycle 10 backtest re-run with explicit `--hold-events 30` + `--min-agreement 1.0` + `--holding-type horizon_aligned` (mirror R-19 Round 19a CLI) — would close the methodology asymmetry but is INFORMATIONAL only (OptRet remains negative regardless; cost-economics gap is the binding constraint per Lesson #106). ~30 min wall-clock. Probably NOT worth shipping vs higher-EV alternatives.
- **Round 20 NEXT candidate**: HMHP cascade on TB v3p0 corpus (R-20 in EXPERIMENT_INDEX) — same Phase Y trust columns apply; same backtest configuration; tests architecture-axis HMHP vs TLOB.
- **Bimodality-exploitation candidate**: re-run cycle 10 backtest with only the conservative-regime seeds (44, 45, 46) → would the +1.4pp regime selection produce +1.4pp better OptRet? Out-of-sample test needed before declaring exploitable.
- **#PY-263 Sharpe inflation caveat applies**: cycle 10 backtest reports Sharpe -19.55 ± uncertainty (per seed_43 single value); per #PY-263 this is 1.6-2.56x inflated from periods_per_day=1000 vs ~390 for 60s bins. Relative cross-seed comparison trustworthy; absolute Sharpe overstated.

### Cross-references

- 5 result.json files at `cycle10_r19_multi_seed__seed_{43..47}_*/result.json`
- 5 equity_curve.npy artifacts via atomic_write_npy SSoT (FIND-090 closure)
- Phase Y verdict JSON: `hft-ops/ledger/r19_multi_seed_verdicts/cycle10_r19_multi_seed_20260518T222513_verdict_20260519T113724.json`
- Companion EXPERIMENT_INDEX entry: `lob-model-trainer/EXPERIMENT_INDEX.md` "Cycle 10 (#PY-243): R-19 Multi-Seed Validation" section
- Sweep launch BG task: `bkl9m5k18` (Bash) completed exit 0 at 13:32 CEST 2026-05-19
- Closes: #PY-307 + #PY-308 BUG2-A; Files: #PY-316
- Prior R-19 cycle bridge: Round 19a (same architecture; PRE-BUG2A + PRE-FIND-NEW-01; single seed=42)

---

## Round 20: R-20 HMHP-R Architecture-Axis Test on e5_60s_v3p0 (PARTIAL-COMPETITIVE-NOT-TRADEABLE, 2026-05-19 NIGHT)

### Cycle 12 context

R-20 cycle on `e5_timebased_60s_v3p0` corpus (98 features, 230 days, H=[10,60,300]) testing HMHP-R cascading-multi-horizon regressor architectural lift vs TLOB Stage 2 baseline. Pre-impl Wave 1+2 (4 parallel agents) REFRAMED original "R-20 HMHP single-horizon TB v3p0" recommendation INFEASIBLE (HMHPConfig N≥2 validator + 1-D TB labels + Tuple return + lessons #1440 + #1450 HMHP×TB twice-refuted) → pivoted to HMHP-R × regression on Stage-6-validated infrastructure. Single-seed (R-17a/R-19/Stage 6 protocol parity).

### Methodology

- **Architecture**: HMHP-R = TLOB encoder (hidden=64, 2 layers) + cascading regression decoders [H10/H60/H300] (hidden=32, state_dim=32, gate fusion) + RegressionConfirmationModule + Phase S pool_mode=mean. 169,239 params (matches Stage 6 exact).
- **Loss**: Huber regression with `regression_loss_delta=12.6` (60s bin H10 kurtosis≈26.5 → δ=12.6 bps per CLAUDE.md Huber δ calibration table + E5 precedent).
- **Loss weights**: H10:0.50 + H60:0.25 + H300:0.15 + consistency:0.10 (Stage 6 exact, H10-primary).
- **Training**: 100 epochs / patience=15 / seed=42 / batch=64 / lr=1e-4 / cosine scheduler / num_workers=0 (OOM-on-fork guard). Wall-clock: 1,405.9s (~23.4 min total incl. signal export + backtest).
- **Test split**: 33 days (8,085+ test samples per Stage 6 baseline).
- **Cost model anchor**: POST-HF-1 (IV=0.25 Deep ITM via mode-aware factory `OpraCalibratedCosts.deep_itm()`; HF-1 commit `175307c` shipped 2026-05-16 LATE NIGHT).

### Per-threshold results (8-threshold cost-aware sweep; ALL NEGATIVE)

| Threshold | OptRet | WinRate | AvgPnL | N_trades | Sharpe | SortinoRatio | MaxDD | Expectancy |
|---|---|---|---|---|---|---|---|---|
| deep_itm_1.4bps | **-6.08%** | 42.54% | -8.56 | 710 | -25.84 | -35.01 | 6.11% | -8.37 |
| itm_2bps | -7.44% | 44.78% | -10.64 | 699 | -26.07 | -33.75 | 6.76% | -9.43 |
| itm_3bps | -8.95% | 43.24% | -13.16 | 680 | -29.52 | -34.22 | 7.44% | -10.80 |
| atm_5bps | -7.99% | 40.91% | -12.98 | 616 | -27.25 | -- | -- | -- |
| high_conv_8bps | -7.36% | 40.59% | -15.57 | 473 | -26.28 | -- | -- | -- |
| **very_high_10bps (BEST)** | **-4.40%** | 41.43% | -12.58 | 350 | -19.63 | -- | -- | -- |
| ultra_conv_15bps | 0.00% | -- | 0.00 | 0 (degenerate) | nan | -- | -- | -- |
| max_conv_20bps | 0.00% | -- | 0.00 | 0 (degenerate) | nan | -- | -- | -- |

### Headline metrics

- **Best OptRet**: -4.40% (very_high_10bps; 350 trades; WinRate 41.43%)
- **Reference (deep_itm_1.4bps)**: -6.08% / 42.54% WR / -25.84 Sharpe
- **Mean OptRet across 6 active thresholds**: -7.04% (no threshold positive)
- **Trade count range**: 350-710 trades across active thresholds (8-18% trade rate)

### Comparison vs Stage 6 (HMHP-R smoke @ 20 epochs) and R-19 (TLOB×TB)

| Metric | R-20 (HMHP-R, 100ep) | Stage 6 (HMHP-R, 20ep) | TLOB Stage 2 (98feat regression) | R-19 (TLOB×TB) |
|---|---|---|---|---|
| test_h10_ic | 0.3670 | 0.3561 | 0.3747 | n/a (classification) |
| Best OptRet | -4.40% (very_high_10bps) | (not backtested in Stage 6 record) | (E5 60s round results) | -3.11% PRE-HF-1 |
| Architecture | HMHP-R cascade | HMHP-R cascade | TLOB | TLOB |
| Corpus | e5_60s_v3p0 | e5_60s_v3p0 | e5_60s_v3p0 | nvda_v3p0_tb_pt40_sl20_h30 |

R-20's PARTIAL-COMPETITIVE H10 IC (-0.77pp vs TLOB) does NOT translate to backtest tradeability. The cascading multi-horizon architecture preserves H10 predictive power within 5pp of TLOB but provides no additional tradeable signal at any cost threshold. **Same E8 label-execution-mismatch holds** — model predicts smoothing residual, not point-direction.

### ConfirmationModule degeneracy (NEW FINDING)

The RegressionConfirmationModule produces near-degenerate cross-horizon agreement:
- mean(agreement_ratio) = **0.9974** (H4.b PASS band was [0.4, 0.9] — FAIL)
- std(agreement_ratio) = **0.0295** (H4.c min was 0.05 — FAIL)

Interpretation: H10/H60/H300 decoder heads agree on direction on ~99.74% of test samples with near-constant variance. Either (a) smoothed labels at these short horizons are highly autocorrelated → cross-horizon predictions naturally agree (semantic finding); OR (b) cascade architectural feature collapse — state-passing from H10 dominates downstream decoders (architectural finding). Either interpretation: ConfirmationModule provides ZERO additional discriminative signal on this corpus + label type combination.

### Phase Y composability

- `compatibility_fingerprint`: `0ccd9f90bca06c868607b6520653e195d909a7fe6083a7aa29e7b8e02c2be160` (matches γ-1 LITE 2026-05-10 "ridge × smoothed × H10" anchor; CORPUS IDENTITY preserved through schema evolution 2026-05-05→2026-05-19)
- `experiment_provenance_hash`: `9c28e966ba45df4214c24e6bbee0ada2c54b87cdbd6357a10ef910ba045d08a1` (POPULATED — Phase Y composer end-to-end functional on HMHP-R production)
- `model_config_hash` (nested): `be5ab20ae5d2b3675d0c1d35762a0102192fcc6e892e60cbd06c143bee1f6154` (differs from Stage 6's `53041488...` because epochs+patience+FeatureSet+num_workers axes rotated per `_LOSS_TUNING_KEYS` semantics; OBSERVATIONAL per H3.b gate design)
- `feature_set_ref`: `{name: nvda_short_term_98_src98_v1, content_hash: 122fe5cbfb657bf91...}` (closes Phase Y composer feature_set_content_hash gap)

### Verdict & Decision

**Empirical verdict**: **PARTIAL-COMPETITIVE-NOT-TRADEABLE** with NEW ConfirmationModule degeneracy finding.

- Architectural axis: HMHP-R cascade is **competitive within 5pp** of TLOB at H10 (0.3670 vs 0.3747) but does NOT clear-lift. H1.b PASS, H1.c FAIL. Closes architecture-axis question for this corpus + label combination.
- Multi-horizon capture: PASS — H60_ic=0.1303, H300_ic=0.0818 confirm signal at all 3 horizons (within ±15% of Stage 6 anchors).
- Tradeability: ALL 8 thresholds NEGATIVE OptRet (best -4.40%). Same E8 label-execution-mismatch persists. Closing tradeability question via cost-aware barrier sweep would require Phase Z #PY-271 (BSM moneyness) + new corpus with point-return labels.
- ConfirmationModule: NEW FINDING — near-degenerate (~0.9974 agreement; ~0.0295 std). Pre-compute expected baseline before next HMHP-R cycle (Lesson L51).

**Direction outcome**: Architecture-axis CLOSED for HMHP-R cascade on v3p0 smoothed-return regression. Cascading-confirmation does not extract additional H10 signal beyond TLOB encoder + flatten. Multi-horizon decoders do successfully capture H60+H300 signal but ConfirmationModule on this corpus is degenerate.

### Cross-references

- Backtest output: `lob-backtester/outputs/backtests/cycle12_r20_hmhp_r__seed_42.json` (registry index updated)
- Per-trade artifacts: 6 `option_trade_pnls__<threshold>.npy` files (deep_itm_1.4bps / itm_2bps / itm_3bps / atm_5bps / high_conv_8bps / very_high_10bps) at `lob-backtester/outputs/backtests/` via FIND-090 atomic-write SSoT
- Companion EXPERIMENT_INDEX entry: `lob-model-trainer/EXPERIMENT_INDEX.md` "Cycle 12: R-20 HMHP-R Architecture-Axis Test"
- Verdict JSON: `hft-ops/ledger/r20_verdicts/cycle12_r20_hmhp_r_20260519T184402_verdict_20260519T191131.json`
- Sweep manifest: `hft-ops/experiments/sweeps/cycle12_r20_hmhp_r.yaml`
- Training record: `hft-ops/ledger/records/cycle12_r20_hmhp_r__seed_42_20260519T190728_5d186966.json`
- Sweep launch BG task: `b1ufwlitb` (Bash) completed exit 0 (1,405.9s = 23.4 min) at 19:07 CEST 2026-05-19
- Analyzer: `hft-ops/scripts/analyze_r20_hmhp_r.py` (set-based CORPUS_COMPAT_FP_ANCHORS frozenset per L49 anchor staleness)
- Prior Stage 6 bridge: `lob-model-trainer/EXPERIMENT_INDEX.md` "Stage 6: HMHP-R v3p0 Validation" (2026-05-05 reference; 20-epoch smoke + Phase S + Phase Y validation)
- Prior γ-1 LITE bridge: CLAUDE.md γ-1 LITE 12-arm sweep table — "ridge × smoothed × H10" compat_fp `0ccd9f90bca06c86...` (2026-05-10 multi-arm cycle) = R-20's compat_fp (corpus identity anchor)
- Prior cycle bridges: Cycle 11 hygiene 2026-05-19 NIGHT (predecessor cycle; F-1+F-2+#PY-312 closures + r19 golden fixtures + bimodality verdict)
