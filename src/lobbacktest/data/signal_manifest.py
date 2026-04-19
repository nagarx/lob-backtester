"""Signal manifest re-export shim (Phase 6 6B.5, 2026-04-17).

The authoritative source lives in ``hft_contracts.signal_manifest``. This
module exists solely for backward compatibility with pre-6B.5 imports:

    from lobbacktest.data.signal_manifest import SignalManifest    # still works
    from lobbacktest.data.signal_manifest import ContractError     # still works
    from lobbacktest.data.signal_manifest import _CONTENT_HASH_RE  # still works

New code should import from hft_contracts:

    from hft_contracts.signal_manifest import SignalManifest, ContractError

Why the move? SignalManifest is a cross-module contract surface: the
trainer's SignalExporter WRITES signal_metadata.json, the backtester
reads + validates it, and the hft-ops signal_export stage harvests the
embedded ``feature_set_ref`` (Phase 4 Batch 4c.4). Placing the contract
schema alongside its peers (``FeatureSet``, ``Provenance``,
``LabelContract``, ``canonical_hash``) on the hft-contracts plane keeps
producer/consumer agreement on the JSON shape in a single authoritative
location.

Phase 6 post-validation (2026-04-18): lazy ``__getattr__`` emits
DeprecationWarning once per symbol access. Scheduled for removal in
on ``_REMOVAL_DATE`` (``2026-10-31``).

See PIPELINE_ARCHITECTURE.md §17.3 producer→consumer matrix.
"""

from __future__ import annotations

import warnings as _warnings

_CANONICAL_MODULE = "hft_contracts.signal_manifest"
# Calendar-driven shim deadline (Phase 7 post-validation, 2026-04-19).
# See hft-ops/src/hft_ops/provenance/lineage.py for rationale — 6 months
# from Phase 6 6B.5.
_REMOVAL_DATE = "2026-10-31"
_PUBLIC_NAMES = frozenset({
    "_CONTENT_HASH_RE",
    "ALIGNED_FILES",
    "CLASSIFICATION_OPTIONAL",
    "CLASSIFICATION_REQUIRED",
    "ContractError",
    "HYBRID_OPTIONAL",
    "HYBRID_REQUIRED",
    "REGRESSION_OPTIONAL",
    "REGRESSION_REQUIRED",
    "SignalManifest",
})
_WARNED: set[str] = set()


def __getattr__(name: str):
    """Lazy re-export with one-time DeprecationWarning per symbol."""
    if name in _PUBLIC_NAMES:
        if name not in _WARNED:
            _WARNED.add(name)
            _warnings.warn(
                f"`lobbacktest.data.signal_manifest.{name}` is a Phase 6 "
                f"6B.5 re-export shim. Migrate to "
                f"`from {_CANONICAL_MODULE} import {name}` before the "
                f"{_REMOVAL_DATE} removal deadline. "
                f"(This warning fires once per symbol per process.)",
                DeprecationWarning,
                stacklevel=2,
            )
        import importlib
        return getattr(importlib.import_module(_CANONICAL_MODULE), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = sorted(_PUBLIC_NAMES)
