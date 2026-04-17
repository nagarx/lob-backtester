"""Phase 4 Batch 4c.4: backtester SignalManifest.feature_set_ref reader.

Locks:
1. signal_metadata.json with feature_set_ref → dataclass surfaces the dict.
2. Legacy signal_metadata.json without the field → None (backward compat).
3. Invalid shape (missing name or content_hash) → None, no crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobbacktest.data.signal_manifest import SignalManifest


def _write_meta(signal_dir: Path, meta: dict) -> None:
    signal_dir.mkdir(parents=True, exist_ok=True)
    (signal_dir / "signal_metadata.json").write_text(json.dumps(meta))
    # SignalManifest._detect_signal_type needs some .npy; drop an empty marker
    (signal_dir / "predicted_returns.npy").write_bytes(b"\x93NUMPY\x01\x00")


class TestFeatureSetRefRead:
    def test_ref_present_in_metadata_surfaces_on_dataclass(self, tmp_path: Path):
        ref = {"name": "x_v1", "content_hash": "a" * 64}
        meta = {
            "signal_type": "regression",
            "model_type": "tlob",
            "split": "test",
            "total_samples": 1000,
            "horizons": [10],
            "exported_at": "2026-04-16T00:00:00+00:00",
            "feature_set_ref": ref,
        }
        _write_meta(tmp_path / "sigs", meta)

        manifest = SignalManifest.from_signal_dir(tmp_path / "sigs")
        assert manifest.feature_set_ref == ref

    def test_legacy_metadata_without_ref_is_none(self, tmp_path: Path):
        meta = {
            "signal_type": "regression",
            "model_type": "tlob",
            "split": "test",
            "total_samples": 1000,
            "horizons": [10],
            "exported_at": "2026-04-16T00:00:00+00:00",
            # NO feature_set_ref — legacy pre-4c.4 signal export
        }
        _write_meta(tmp_path / "legacy", meta)

        manifest = SignalManifest.from_signal_dir(tmp_path / "legacy")
        assert manifest.feature_set_ref is None


class TestInvalidShapeGracefullyIgnored:
    def test_missing_content_hash_is_none(self, tmp_path: Path):
        meta = {
            "signal_type": "regression",
            "model_type": "tlob",
            "split": "test",
            "total_samples": 1000,
            "horizons": [10],
            "feature_set_ref": {"name": "x"},  # no content_hash
        }
        _write_meta(tmp_path / "bad1", meta)
        manifest = SignalManifest.from_signal_dir(tmp_path / "bad1")
        assert manifest.feature_set_ref is None, (
            "Missing content_hash → None; backtester never crashes on "
            "malformed-but-present field."
        )

    def test_wrong_type_is_none(self, tmp_path: Path):
        meta = {
            "signal_type": "regression",
            "model_type": "tlob",
            "split": "test",
            "total_samples": 1000,
            "horizons": [10],
            "feature_set_ref": "not_a_dict",  # wrong type
        }
        _write_meta(tmp_path / "bad2", meta)
        manifest = SignalManifest.from_signal_dir(tmp_path / "bad2")
        assert manifest.feature_set_ref is None
