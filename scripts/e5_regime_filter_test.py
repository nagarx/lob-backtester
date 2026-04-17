#!/usr/bin/env python3
"""
E5 Phase A: Regime-Filtered Backtest Validation.

Tests whether filtering to specific intraday windows improves backtesting results.
Uses position-based time estimation (NOT time_regime feature index 93, which has
C2/DST bug at 60s bins — see CLAUDE.md Known Issues).

At 60s bins with grid-aligned sampling starting at 9:30 ET:
- First valid sequence starts at ~10:30 ET (after 20-min window + OFI warmup)
- Each subsequent sample = +60 seconds
- Sample position i ≈ 10:30 + i minutes ET

Profiler-validated optimal windows (233 days XNAS):
- Afternoon (14:00-15:30 ET): OFI-return r=0.653 (best)
- Midday (12:00-14:00 ET): OFI-return r=0.599
- Morning (10:00-12:00 ET): OFI-return r=0.582
- Open (9:30-10:00 ET): OFI-return r=0.504 (worst, AVOID)

Usage:
    python scripts/e5_regime_filter_test.py \
        --signals ../lob-model-trainer/outputs/experiments/e5_60s_huber_nocvml/signals/test/ \
        --sequences ../data/exports/e5_timebased_60s/test/ \
        --output-dir outputs/backtests/e5_round7b/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def estimate_wall_clock_minute(sample_idx: int, day_n_samples: int) -> float:
    """Estimate minutes since market open (9:30 ET) for a sample position.

    At 60s bins, grid-aligned to 9:30 ET:
    - First bin = 9:31 ET (first event after open + 60s)
    - With window=20: first sequence starts at bin 20 = 9:50 ET
    - With OFI warmup (~10 bins): first valid sample ≈ bin 30 = 10:00 ET
    - Each sample = +1 minute

    Args:
        sample_idx: Position in day (0-based)
        day_n_samples: Total samples this day

    Returns:
        Approximate minutes since 9:30 ET
    """
    # First valid sample ≈ 30 minutes after 9:30 = 10:00 ET
    first_sample_minute = 30
    return first_sample_minute + sample_idx


def create_regime_mask(
    sequences_dir: Path,
    n_signals: int,
    window_start_min: float = 0,
    window_end_min: float = 390,
) -> np.ndarray:
    """Create boolean mask for samples within the specified time window.

    Args:
        sequences_dir: Directory with {day}_sequences.npy files
        n_signals: Total number of signal samples (must match)
        window_start_min: Minutes after 9:30 ET to start (e.g., 270 = 14:00)
        window_end_min: Minutes after 9:30 ET to end (e.g., 360 = 15:30)

    Returns:
        Boolean array [n_signals] where True = within time window
    """
    seq_files = sorted(sequences_dir.glob("*_sequences.npy"))
    mask_parts = []

    for sf in seq_files:
        seqs = np.load(sf, mmap_mode="r")
        n_day = seqs.shape[0]

        day_mask = np.zeros(n_day, dtype=bool)
        for i in range(n_day):
            minute = estimate_wall_clock_minute(i, n_day)
            day_mask[i] = window_start_min <= minute < window_end_min

        mask_parts.append(day_mask)

    full_mask = np.concatenate(mask_parts)
    if len(full_mask) != n_signals:
        raise ValueError(
            f"Mask length {len(full_mask)} != signal length {n_signals}. "
            f"Sequence files may not match signal export."
        )
    return full_mask


def filter_and_save_signals(
    signal_dir: Path,
    output_dir: Path,
    mask: np.ndarray,
    window_name: str,
) -> dict:
    """Apply mask to signal files and save filtered versions.

    Returns metadata dict with counts and filter stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_files = [
        "predicted_returns.npy",
        "regression_labels.npy",
        "spreads.npy",
        "prices.npy",
    ]

    stats = {"window": window_name, "total": int(len(mask)), "selected": int(mask.sum())}
    stats["filter_rate"] = stats["selected"] / stats["total"]

    for fname in signal_files:
        fpath = signal_dir / fname
        if fpath.exists():
            arr = np.load(fpath)
            filtered = arr[mask]
            np.save(output_dir / fname, filtered)
            stats[fname] = {"original": len(arr), "filtered": len(filtered)}

    # Copy and update metadata
    meta_path = signal_dir / "signal_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["regime_filter"] = stats
        meta["n_samples_filtered"] = stats["selected"]
        with open(output_dir / "signal_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    return stats


def main():
    parser = argparse.ArgumentParser(description="E5 Regime-Filtered Backtest Test")
    parser.add_argument("--signals", required=True, help="Signal export directory")
    parser.add_argument("--sequences", required=True, help="Sequences directory (for time estimation)")
    parser.add_argument("--output-dir", default="outputs/backtests/e5_round7b", help="Output directory")
    args = parser.parse_args()

    signal_dir = Path(args.signals)
    seq_dir = Path(args.sequences)
    output_base = Path(args.output_dir)

    # Load signal count
    pred = np.load(signal_dir / "predicted_returns.npy")
    n_signals = len(pred)
    print(f"Total signals: {n_signals}")

    # Define time windows (minutes after 9:30 ET)
    windows = {
        "afternoon_14_1530": (270, 360),      # 14:00-15:30 ET (profiler BEST)
        "midday_1200_1400": (150, 270),        # 12:00-14:00 ET
        "core_1100_1500": (90, 330),           # 11:00-15:00 ET (broad prime)
        "prime_1200_1530": (150, 360),          # 12:00-15:30 ET (extended prime)
    }

    for window_name, (start_min, end_min) in windows.items():
        start_time = f"{9 + start_min // 60}:{start_min % 60:02d}"
        end_time = f"{9 + end_min // 60}:{end_min % 60:02d}"
        print(f"\n{'='*60}")
        print(f"  Window: {window_name} ({start_time}-{end_time} ET)")
        print(f"{'='*60}")

        mask = create_regime_mask(seq_dir, n_signals, start_min, end_min)
        n_selected = mask.sum()
        print(f"  Samples: {n_selected}/{n_signals} ({n_selected/n_signals*100:.1f}%)")

        if n_selected < 50:
            print(f"  SKIP: Too few samples ({n_selected})")
            continue

        # Filter and save
        out_dir = output_base / window_name
        stats = filter_and_save_signals(signal_dir, out_dir, mask, window_name)
        print(f"  Filtered signals saved to: {out_dir}")
        print(f"  Filter rate: {stats['filter_rate']:.1%}")

    print(f"\n{'='*60}")
    print(f"  DONE — Run backtests on each filtered directory:")
    for window_name in windows:
        out_dir = output_base / window_name
        print(f"    python scripts/run_regression_backtest.py \\")
        print(f"      --signals {out_dir} \\")
        print(f"      --name e5_r7b_{window_name} --deep-itm --hold-events 10")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
