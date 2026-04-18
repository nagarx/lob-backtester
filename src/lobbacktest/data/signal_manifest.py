"""Signal manifest re-export shim (Phase 6 6B.5, 2026-04-17).

The authoritative source lives in `hft_contracts.signal_manifest`. This
module exists solely for backward compatibility with pre-6B.5 imports:

    from lobbacktest.data.signal_manifest import SignalManifest   # still works
    from lobbacktest.data.signal_manifest import ContractError    # still works

New code should import from hft_contracts:

    from hft_contracts.signal_manifest import SignalManifest, ContractError

Why the move? SignalManifest is a cross-module contract surface: the
trainer's SignalExporter WRITES signal_metadata.json, the backtester
reads + validates it, and the hft-ops signal_export stage harvests the
embedded `feature_set_ref` (Phase 4 Batch 4c.4). Placing the contract
schema alongside its peers (`FeatureSet`, `Provenance`, `LabelContract`,
`canonical_hash`) on the hft-contracts plane keeps producer/consumer
agreement on the JSON shape in a single authoritative location.

See PIPELINE_ARCHITECTURE.md §17.3 producer→consumer matrix.
"""

from __future__ import annotations

from hft_contracts.signal_manifest import (
    _CONTENT_HASH_RE,
    ALIGNED_FILES,
    CLASSIFICATION_OPTIONAL,
    CLASSIFICATION_REQUIRED,
    ContractError,
    HYBRID_OPTIONAL,
    HYBRID_REQUIRED,
    REGRESSION_OPTIONAL,
    REGRESSION_REQUIRED,
    SignalManifest,
)

__all__ = [
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
]
