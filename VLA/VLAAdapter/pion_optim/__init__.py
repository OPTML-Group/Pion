"""Pion optimizers vendored for the VLA-Adapter sub-repo.

Exposes the three base optimizers used by ``vla-scripts/finetune_pion.py``:

  * ``Muon``         -- Newton-Schulz orthogonalization (Muon baseline).
  * ``DefaultPion``  -- Pion with the two-stage Promotion + Suppression
                        (high-pass NS) iteration applied to the whole matrix.
  * ``LowRankMuon``  -- Muon variant that uses exact SVD to project the
                        update onto the top-``k`` singular subspace before
                        orthogonalization.

Auxiliary 0/1-D parameters and embedding / output-head weights should be
routed to ``torch.optim.AdamW`` instead of these classes.
"""

from .muon import (
    fast_newtonschulz,
    high_pass_ns,
    Muon,
    DefaultPion,
    LowRankMuon,
)

__all__ = [
    "fast_newtonschulz",
    "high_pass_ns",
    "Muon",
    "DefaultPion",
    "LowRankMuon",
]
