from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t


def total_scale(cov_reported: np.ndarray, tau: float, lam: float) -> np.ndarray:
    """Return the Student-t scale/shape matrix: lambda^2*S + tau^2*I.

    For nu > 2, its ordinary covariance is nu/(nu-2) times this matrix.
    """
    cov = np.asarray(cov_reported, dtype=float)
    return lam**2 * cov + tau**2 * np.eye(cov.shape[0])


def mvt_logpdf(y: np.ndarray, mean: np.ndarray, shape: np.ndarray, nu: float) -> float:
    """Log-density using SciPy's multivariate-t *shape* matrix.

    The corresponding covariance is nu/(nu-2) times ``shape`` for nu>2;
    throughout the Student-t pathway the matrix supplied here is therefore a
    scale/shape parameter, not the ordinary covariance matrix.
    """
    return float(multivariate_t.logpdf(np.asarray(y,float), loc=np.asarray(mean,float), shape=np.asarray(shape,float), df=float(nu)))


def conditional_mvt(y_train, mean_train, mean_hold, scale_train, scale_hold, scale_cross, nu):
    y_train=np.asarray(y_train,float); mt=np.asarray(mean_train,float); mh=np.asarray(mean_hold,float)
    A=np.asarray(scale_train,float); C=np.asarray(scale_hold,float); B=np.asarray(scale_cross,float)
    delta=y_train-mt
    Ainv_delta=np.linalg.solve(A,delta)
    loc=mh + B.T@Ainv_delta
    schur=C-B.T@np.linalg.solve(A,B)
    schur=(schur+schur.T)/2
    q=float(delta@Ainv_delta)
    df=float(nu+len(y_train))
    scale=((nu+q)/df)*schur
    return loc, scale, df


def posterior_predictive_lppd(logpdf_draws) -> float:
    x=np.asarray(logpdf_draws,float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError('logpdf_draws must contain finite values')
    return float(logsumexp(x)-np.log(x.size))
