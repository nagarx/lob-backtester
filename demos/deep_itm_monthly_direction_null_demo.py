# PEDAGOGICAL DEMO — not an experiment, not production infra.
"""Deep-ITM ~1-month option held intraday on the real NVDA Ridge signal = E[net] < 0.

Companion to tests/test_engine/test_deep_itm_monthly_demo.py (the rigorous proof).
This runs the theorem on REAL data: the near-null NVDA Temporal-Ridge H10 signal
(the universality / FINDING-002 model whose point-return directional skill is ~0).

It shows: a deep-ITM (delta~1) ~1-month call/put, entered in the MODEL's predicted
direction and held one intraday step, has gross P&L ~= delta * (signed point return),
whose mean is ~0 because the model has no point-return direction edge — so net of the
IBKR-calibrated 1.4 bps deep-ITM round-trip cost, E[P&L] < 0. The option wrapper adds
cost/leverage, never signal (FINDING-002 / FINDING-047).

NOT an edge claim (any_tradeable_edge = False by construction). Uses POINT returns
(prices), never the smoothed regression_labels (E8 trap, FINDING-001/048). Cross-day
steps are excluded by the KNOWN DAY STRUCTURE (the test split is 33 days x 245
samples/day = 8085; the step into each new day is overnight) — NOT by a magnitude
tripwire, which a metrics-validator showed misses these gaps (they run 19-620 bps,
overlapping intraday noise). They are not valid no-overnight intraday trades; the
contaminating overnight steps alone carried +83 bps gross, so excluding them makes
the null CLEANER (intraday gross ~ -0.11 bps vs the contaminated +0.22). Robustness:
drop-top-K (FINDING-005). Run:

    lob-backtester/.venv/bin/python demos/deep_itm_monthly_direction_null_demo.py
"""
from __future__ import annotations

import os

import numpy as np

from lobbacktest.engine.option_pricing import bs_delta

_SIG = (
    "/Users/knight/code_local/HFT-pipeline-v2/lob-model-trainer/outputs/"
    "experiments/nvda_temporal_ridge_h10_e5_60s_v3p0/signals/test"
)
_DEEP_ITM_COST_BPS = 1.4          # IBKR 316-fill deep-ITM round-trip breakeven
_SAMPLES_PER_DAY = 245             # test split = 33 days x 245 (validator-confirmed from
                                   # data/exports/e5_timebased_60s_v3p0/test/); the step
                                   # into each new day is overnight (no-overnight: exclude)
_DATA_ERR_TRIPWIRE_BPS = 1000.0    # secondary guard: |1-step ret|>10% => bad print, exclude
_MONEYNESS = 0.85                  # deep-ITM call strike = 0.85 * S
_TAU_1M = 21.0 / 252.0
_SIGMA, _R = 0.25, 0.0


def _summary(tag: str, net_bps: np.ndarray) -> None:
    n = net_bps.size
    print(
        f"  {tag:<22} n={n:>5}  mean_net={net_bps.mean():+.3f} bps  "
        f"median={np.median(net_bps):+.3f}  win_rate={(net_bps > 0).mean():.3f}"
    )


def main() -> None:
    prices = np.load(os.path.join(_SIG, "prices.npy"))
    pred = np.load(os.path.join(_SIG, "predicted_returns.npy"))
    if pred.ndim == 2:
        pred = pred[:, 0]
    if prices.size % _SAMPLES_PER_DAY != 0:
        raise ValueError(
            f"N={prices.size} not divisible by SAMPLES_PER_DAY={_SAMPLES_PER_DAY}; "
            "day structure unknown for this signal — set _SAMPLES_PER_DAY or the "
            "overnight exclusion is wrong (fail-loud, hft-rules s.8)."
        )

    # 1-step POINT return in bps (execution-aligned; NOT the smoothed label)
    ret_bps = (prices[1:] / prices[:-1] - 1.0) * 1e4
    direction = np.sign(pred[:-1])  # the model's predicted direction at entry

    # deep-ITM delta at the sample-mean price (honest: computed, not assumed ~1)
    s_bar = float(np.mean(prices))
    delta = bs_delta(True, s_bar, _MONEYNESS * s_bar, _TAU_1M, _R, _SIGMA)

    # gross option P&L (bps of underlying) ~= delta * signed point return
    gross_bps = delta * direction * ret_bps
    traded = direction != 0.0  # only directional signals are trades

    # overnight = the step INTO each new day (idx i is last-of-day iff (i+1)%SPD==0),
    # excluded by the KNOWN day structure (NOT a magnitude tripwire — validator-fixed).
    step_idx = np.arange(ret_bps.size)
    overnight = ((step_idx + 1) % _SAMPLES_PER_DAY) == 0
    data_err = np.abs(ret_bps) > _DATA_ERR_TRIPWIRE_BPS  # secondary bad-print guard
    intraday = traded & ~overnight & ~data_err

    print("=" * 78)
    print("DEEP-ITM ~1-MONTH OPTION HELD INTRADAY — direction-null demonstration")
    print("  signal: NVDA Temporal-Ridge H10 (near-null model, FINDING-002)")
    print(f"  deep-ITM delta (S~{s_bar:.1f}, K=0.85S, tau=21/252, sigma=0.25) = {delta:.4f}")
    print(f"  round-trip cost = {_DEEP_ITM_COST_BPS} bps (IBKR deep-ITM breakeven)")
    print(
        f"  excluded {int((overnight & traded).sum())} overnight + "
        f"{int((data_err & traded & ~overnight).sum())} bad-print steps "
        f"({prices.size // _SAMPLES_PER_DAY} days x {_SAMPLES_PER_DAY})"
    )
    print("-" * 78)

    net_all = gross_bps[traded] - _DEEP_ITM_COST_BPS
    net_intra = gross_bps[intraday] - _DEEP_ITM_COST_BPS
    overnight_only = gross_bps[traded & overnight]  # diagnostic: where contamination lived
    # drop-top-K robustness on the intraday set (FINDING-005)
    gi = gross_bps[intraday]
    order = np.argsort(-np.abs(gi))
    keep = np.ones(gi.size, dtype=bool)
    keep[order[:20]] = False
    net_drop20 = gi[keep] - _DEEP_ITM_COST_BPS

    _summary("all (incl. overnight)", net_all)
    _summary("intraday only [HEADLINE]", net_intra)
    _summary("intraday drop-top-20", net_drop20)
    print(
        f"  {'overnight-only (excluded)':<22} n={overnight_only.size:>5}  "
        f"mean_gross={overnight_only.mean():+.3f} bps  "
        f"(NOT tradeable no-overnight; this is where the contaminated +gross lived)"
    )

    print("-" * 78)
    gross_mean = float(gi.mean())
    print(
        f"  model directional GROSS (intraday) = {gross_mean:+.3f} bps/trade  "
        f"<<  {_DEEP_ITM_COST_BPS} bps cost"
    )
    verdict = "E[net] < 0  => direction null confirmed (wrapper adds cost, not signal)"
    if net_intra.mean() >= 0:
        verdict = "UNEXPECTED net>=0 — investigate (drift/leakage?) before trusting"
    print(f"  VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
