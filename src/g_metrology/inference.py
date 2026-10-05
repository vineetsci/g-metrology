from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PosteriorDraws:
    samples: dict[str,np.ndarray]
    logpost: np.ndarray
    acceptance: float
    chains: dict[str,np.ndarray]