from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import yaml

# Frozen hashes for the authoritative input datasets. Input files are required to be byte-identical.
CANONICAL_SOURCE_SHA256 = {
    'codata_2010_g.yaml': '4cf2d12f05bf7e0031054821e7d25066376d87f369d23a52fe568fe833a13276',
    'codata_2018_g.yaml': 'f7769f47dd4db10ba25f9663e921e9f48035ff753e47ad2a277976dad33ef662',
    'nist2026_configurations.yaml': '8d1255f63ec2fc080dbb40f28b30411881695d84c893dd89f56ee612482bbf30',
}

@dataclass(frozen=True)
class Measurement:
    label: str
    value_si: float
    u_si: float
    year: int | None
    method: str = ''
    reference: str = ''
    notes: str = ''
    u_relative: float | None = None
    dark_u_si: float | None = None
    config_effect_si: float | None = None
    configuration: str = ''

@dataclass(frozen=True)
class Dataset:
    name: str
    unit_si: str
    measurements: tuple[Measurement, ...]
    correlations: tuple[tuple[int,int,float], ...]
    source_path: str
    source_sha256: str
    raw_expansion_factor: float | None = None
    auxiliary: dict | None = None

    @property
    def y(self): return np.array([m.value_si for m in self.measurements], dtype=float)
    @property
    def u(self): return np.array([m.u_si for m in self.measurements], dtype=float)
    @property
    def labels(self): return [m.label for m in self.measurements]
    @property
    def years(self): return np.array([m.year if m.year is not None else -1 for m in self.measurements], dtype=int)
    @property
    def methods(self): return [m.method for m in self.measurements]

def load_yaml(path: str | Path) -> Dataset:
    p=Path(path); raw=p.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    expected=CANONICAL_SOURCE_SHA256.get(p.name)
    if expected is not None and digest != expected:
        raise ValueError(f'Frozen source hash mismatch for {p}: expected {expected}, got {digest}')
    cfg=yaml.safe_load(raw.decode('utf-8'))
    ms=[]
    for m in cfg['measurements']:
        label=m['label']
        value_si = float(m['value_SI'])
        if 'u_std_SI' in m:
            u_si = float(m['u_std_SI'])
        elif 'u_relative' in m:
            u_si = value_si * float(m['u_relative'])
        else:
            raise KeyError(f'{label}: expected u_std_SI or u_relative')
        if m.get('year') is None:
            raise KeyError(f'{label}: explicit year is required; silent YEAR fallback is disabled')
        method = m.get('method')
        if not method:
            raise KeyError(f'{label}: explicit method is required; silent METHOD fallback is disabled')
        ms.append(Measurement(
            label=label,
            value_si=value_si,
            u_si=u_si,
            year=int(m['year']),
            method=method,
            reference=m.get('reference',''), notes=m.get('notes',''),
            u_relative=(float(m['u_relative']) if 'u_relative' in m else None),
            dark_u_si=(float(m['dark_u_SI']) if 'dark_u_SI' in m else None),
            config_effect_si=(float(m['config_effect_SI']) if 'config_effect_SI' in m else None),
            configuration=m.get('configuration',''),
        ))
    corrs=tuple((int(c['i']),int(c['j']),float(c['r'])) for c in cfg.get('correlations',[]))
    k=cfg.get('expansion_factor_k')
    auxiliary = {k:v for k,v in cfg.items() if k not in {'dataset','title','unit_si','unit_SI','units','source','doi','measurements','correlations','expansion_factor_k','notes'}}
    if 'notes' in cfg:
        auxiliary['dataset_notes'] = cfg['notes']
    return Dataset(cfg.get('dataset') or cfg.get('title','unknown'), cfg.get('unit_si') or cfg.get('unit_SI') or cfg.get('units',{}).get('value',''), tuple(ms), corrs, str(p), digest, float(k) if k is not None else None, auxiliary or None)


ACTIVE_COVARIANCE_K = 1.0

def assert_active_covariance_k(ds: Dataset, k: float = ACTIVE_COVARIANCE_K) -> None:
    """Enforce the frozen active-analysis convention: stored expansion is metadata; k=1 is active."""
    if not np.isclose(float(k), ACTIVE_COVARIANCE_K):
        raise ValueError(f'Active analysis requires k={ACTIVE_COVARIANCE_K}; requested k={k}')
    if ds.raw_expansion_factor is not None and ds.raw_expansion_factor <= 0:
        raise ValueError(f'Invalid stored expansion factor for {ds.source_path}: {ds.raw_expansion_factor}')

def active_covariance_from_dataset(ds: Dataset) -> np.ndarray:
    assert_active_covariance_k(ds, ACTIVE_COVARIANCE_K)
    return covariance_from_dataset(ds, ACTIVE_COVARIANCE_K)

def covariance_from_dataset(ds: Dataset, k: float = 1.0) -> np.ndarray:
    # The stored 2018 uncertainties do not apply CODATA-2018 (k=3.9).
    # Active reproducibility analyses use k=1.0 and estimate any additional scale
    # through the specified statistical model. Fail loudly if a non-unity scale is requested.
    if ds.raw_expansion_factor is not None and np.isclose(ds.raw_expansion_factor, 3.9) and not np.isclose(float(k), 1.0):
        raise ValueError(f'Active CODATA-2018 analysis requires k=1.0; requested k={k}')
    u=float(k)*ds.u
    S=np.diag(u*u)
    for i,j,r in ds.correlations:
        S[i,j]=S[j,i]=r*u[i]*u[j]
    eig=np.linalg.eigvalsh(S)
    if eig.min() <= 0:
        raise ValueError(f'Covariance not positive definite; min eigenvalue {eig.min():.3g}')
    cond=float(np.linalg.cond(S))
    if not np.isfinite(cond):
        raise ValueError('Covariance condition number is not finite')
    if cond > 1e12:
        import warnings
        warnings.warn(f'Covariance condition number is high: {cond:.3e}', RuntimeWarning)
    return S


def covariance_diagnostics(S: np.ndarray, *, name: str = 'covariance', max_condition: float = 1e10) -> dict:
    """Validate covariance structure and return eigenvalue/conditioning diagnostics."""
    A=np.asarray(S,float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]: raise ValueError(f'{name}: covariance must be square')
    if not np.all(np.isfinite(A)): raise ValueError(f'{name}: covariance contains non-finite values')
    if not np.allclose(A,A.T,rtol=1e-12,atol=1e-18): raise ValueError(f'{name}: covariance is not symmetric')
    eig=np.linalg.eigvalsh(A); mineig=float(eig[0]); maxeig=float(eig[-1])
    if mineig <= 0: raise ValueError(f'{name}: covariance is not positive definite; min eigenvalue {mineig:.3e}')
    cond=maxeig/mineig
    if cond > max_condition: raise ValueError(f'{name}: covariance condition number {cond:.3e} exceeds limit {max_condition:.3e}')
    return {'min_eigenvalue':mineig,'max_eigenvalue':maxeig,'condition_number':float(cond)}