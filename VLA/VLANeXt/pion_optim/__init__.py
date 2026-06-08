"""Pion optimizers vendored for the VLANeXt sub-repo.

Exposes the two base optimizers used by ``scripts/train_pion.py``:

  * ``Muon``         -- Newton-Schulz orthogonalization (Muon baseline).
  * ``DefaultPion``  -- Pion with the two-stage Promotion + Suppression
                        (high-pass NS) iteration applied to the whole matrix.

Auxiliary 0/1-D parameters and embedding / output-head weights should be
routed to ``torch.optim.AdamW`` instead of these classes.
"""

from .muon import (
    fast_newtonschulz,
    high_pass_ns,
    Muon,
    DefaultPion,
)

__all__ = [
    "fast_newtonschulz",
    "high_pass_ns",
    "Muon",
    "DefaultPion",
]
