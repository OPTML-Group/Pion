"""Pion: sPectral hIgh-pass Optimization on momeNtum.

Reference implementation of the optimizers used in

    "Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass
     Remedies for VLA and RLVR"
"""

from .pion import (
    fast_newtonschulz,
    high_pass_ns,
    Muon,
    DefaultPion,
    PerHeadPion,
    LowRankMuon,
)

__all__ = [
    "fast_newtonschulz",
    "high_pass_ns",
    "Muon",
    "DefaultPion",
    "PerHeadPion",
    "LowRankMuon",
]
