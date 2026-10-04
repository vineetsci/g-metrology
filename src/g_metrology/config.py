from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA_DIR=ROOT/'data'; RESULTS_DIR=ROOT/'results'
@dataclass(frozen=True)
class Priors:
    mu_sd: float=5.0
    tau_beta: float=1.0
    log_lambda_mu: float=0.0
    log_lambda_sd: float=1.0
    # nu_minus2 follows an exponential prior with rate 1/28 (scale 28).
    # This is deliberately recorded explicitly because rate-vs-scale notation is easy to misread.
    nu_minus2_rate: float=1/28
DEFAULT_PRIORS=Priors()
SEED=20260903
