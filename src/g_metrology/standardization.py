from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StandardizationContext:
    mean_si: float
    scale_si: float
    @classmethod
    def from_data(cls, x):
        x=np.asarray(x,float)
        if x.ndim != 1 or len(x)<2: raise ValueError('Need at least 2 one-dimensional observations')
        s=float(np.std(x,ddof=0)); m=float(np.mean(x))
        if not np.isfinite(s) or s<=0: raise ValueError('Invalid scale')
        return cls(m,s)
    def transform(self,x): return (np.asarray(x,float)-self.mean_si)/self.scale_si
    def inverse_transform(self,x): return np.asarray(x,float)*self.scale_si+self.mean_si
    def covariance_to_std(self,S): return np.asarray(S,float)/(self.scale_si**2)
    def covariance_to_si(self,S): return np.asarray(S,float)*(self.scale_si**2)
