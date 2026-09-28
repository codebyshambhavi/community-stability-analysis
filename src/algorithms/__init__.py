"""The four LPA-family algorithms under comparison. Every function has the signature
`run_x(graph: GraphBundle, seed: int, ...) -> AlgorithmResult` (see common.py) so the
Stage 5 experiment runner can iterate ALGORITHMS uniformly."""
from __future__ import annotations

from .common import AlgorithmResult
from .flpa import run_flpa
from .lpa import run_lpa
from .semi_sync import run_semi_sync_lpa
from .slpa import run_slpa

ALGORITHMS = {
    "lpa": run_lpa,
    "semi_sync_lpa": run_semi_sync_lpa,
    "flpa": run_flpa,
    "slpa": run_slpa,
}

__all__ = ["AlgorithmResult", "run_lpa", "run_semi_sync_lpa", "run_flpa", "run_slpa", "ALGORITHMS"]
