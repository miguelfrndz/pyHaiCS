"""
This script provides a Python implentation of the CODA library in R.

This implementation aims to replicate the functionality of the CODA library for diagnostic checking of MCMC output. The code available here has been adapted from: https://github.com/aki-nishimura/mcmc-diagnostics
"""

### AR Model Fitting ###

import numpy as np
import math
from scipy.linalg import solve_toeplitz
from scipy.special import kv as modified_bessel_2


def ar_fit(x, max_order=None):

    if max_order is None:
        max_order = min(
            len(x) - 1, math.ceil(10 * np.log10(len(x)))
        )

    aic_best = float('inf')
    ar_order_best = 0
    ar_coef_best = None
    for order in range(max_order + 1):
        ar_coef, aic = ar_fit_via_yule_walker(x, order)
        if aic < aic_best:
            aic_best = aic
            ar_order_best = order
            ar_coef_best = ar_coef

    return ar_order_best, ar_coef_best


def ar_fit_via_yule_walker(x, order, acf_method="mle", demean=True):
    """
    Estimate AR(p) parameters of a sequence x using the Yule-Walker equation.

    Parameters
    ----------
    x : 1d numpy array
    order : integer
        The order of the autoregressive process.
    acf_method : {'unbiased', 'mle'}, optional
       Method can be 'unbiased' or 'mle' and this determines denominator in
       estimating autocorrelation function (ACF) at lag k. If 'mle', the
       denominator is  `n = x.shape[0]`, if 'unbiased' the denominator is `n - k`.
    demean : bool
        True, the mean is subtracted from `x` before estimation.
    """

    if demean:
        x = x.copy()
        x -= x.mean()

    if acf_method == "unbiased":
        denom = lambda lag: len(x) - lag
    else:
        denom = lambda lag: len(x)
    if x.ndim > 1 and x.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")

    auto_cov = np.zeros(order + 1, np.float64)
    auto_cov[0] = (x ** 2).sum() / denom(0)
    for lag in range(1, order + 1):
        auto_cov[lag] = np.sum(x[0:-lag] * x[lag:]) / denom(lag)

    if order == 0:
        ar_coef = None
        innovation_var = auto_cov[0]
    else:
        ar_coef = _solve_yule_walker(auto_cov)
        innovation_var = auto_cov[0] - (auto_cov[1:] * ar_coef).sum()

    aic = compute_aic(innovation_var, order, len(x))

    return ar_coef, aic


def _solve_yule_walker(auto_cov):
    ar_coef = solve_toeplitz(auto_cov[:-1], auto_cov[1:])
    return ar_coef


def compute_aic(innovation_var, ar_order, n_obs):
    # I don't quite understand the formula (the maximized likelihood part), but
    # it agrees with the one used in R's ar.yw.default as well as Python's statsmodels.
    # Also, described in http://pages.stern.nyu.edu/~churvich/TimeSeries/Handouts/AICC.pdf
    return n_obs * np.log(innovation_var) + 2 * (1 + ar_order)

### Cramer von Mises Statistic ###

def is_stationarity(x, signif_level=.05):
    return calculate_p_value(x) < signif_level


def calculate_p_value(x, ess_chain_frac=.5):
    """ Calculate the p-value under stationarity using Cramer-von-Mises statistics.

    Parameters
    ----------
    x: 1d numpy array
    ess_chain_frac : float between 0 and 1
        Fraction of the tail of the chain used in estimating the effective
        sample size, which is needed in computing the test statistics.
    """
    if (type(x) is not np.ndarray) or x.ndim != 1:
        raise TypeError("The input must be a 1d numpy array.")
    cvm_stat = cramer_von_mises_statistic(x, ess_chain_frac)
    p_val = 1 - cramer_von_mises_cdf(cvm_stat)
    return p_val


def cramer_von_mises_statistic(x, ess_chain_frac=.5):
    bb_stat = _brownian_bridge_statistic(x, 1 - ess_chain_frac)
    cvm_stat = np.trapz(bb_stat ** 2, np.linspace(0, 1, len(bb_stat)))
    return cvm_stat


def _brownian_bridge_statistic(x, frac_discard=.5):
    """
    Parameters
    ----------
    frac_discard : float in [0, 1]
        The non-stationarity can inflate the estimate of the spectrum at zero.
        To avoid this, as proposed in Heidelberger and Welch (1983), discard
        the initial fraction of the time series `x` (and hope that the rest of
        the sequence looks more stationary).
    """
    n_discard = math.ceil(frac_discard * len(x))
    x_subseq = x[n_discard:]
    spectrum_at_zero = np.var(x_subseq) / estimate_ess(x_subseq, normed=True)
    cumsum = np.concatenate(([0], np.cumsum(x)))
    linear_interp = np.arange(len(x) + 1) * np.mean(x)
    stat = (cumsum - linear_interp) / np.sqrt(len(x) * spectrum_at_zero)
    return stat


def cramer_von_mises_cdf(x, n_summand=None):
    """
    Computes the cumulative distribution function of the (asymptotic)
    Cramer-von-Mises statistics (the integral of a squared Brownian bridge
    process) via a series expansion formula.

    Parameters
    ----------
    x : scalar
    n_summand : optional, int
        The R coda (as of ver 0.19-1) uses 4, but we need more terms for a large
        value of `x`.
    """
    if n_summand is None:
        n_summand = 5 if x < 10 else 10
    cum_density = 0
    for k in range(n_summand):
        cum_density += _cvm_cdf_summand(x, k)
    return cum_density


def _cvm_cdf_summand(x, k):
    temp = (4 * k + 1) ** 2 / 16 / x
    kth_summand = (
        1 / math.pi ** (3 / 2) / math.sqrt(x)
        * math.gamma(k + .5) / math.factorial(k) * math.sqrt(4 * k + 1)
        * math.exp(-temp) * modified_bessel_2(.25, temp)
    ) # Equation (1.3) in Csorgo and Faraway (1996).
    return kth_summand

### Effective Sample Size ###
def ar_process_fit(samples, axis=0, normed=False, max_ar_order=None):
    """
    Estimates effective sample sizes of samples along the specified axis by
    fitting an autoregressive process via the Yule-Walker equation. The order
    of the AR process is determined via AIC.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    series_length = samples.shape[axis]
    if max_ar_order is None:
        if series_length <= 100:
            max_ar_order = math.ceil(series_length / 5)
        else:
            max_ar_order = math.ceil(10 * np.log10(series_length))

    n_param = samples.shape[1 - axis]

    if series_length == 1: # Edge case
        ess = np.zeros(n_param)
    else:
        if axis == 0:
            samples = samples.T
        ess = np.array([
            _ar_process_fit_1d(x, max_ar_order) for x in samples
            # Loop is over the rows of samples.
        ])

    if normed: ess /= series_length

    return ess

def _ar_process_fit_1d(x, max_ar_order):

    ar_order, ar_coef = ar_fit(x, max_ar_order)

    if ar_order == 0:
        auto_corr_time = 1
    else:
        x_std = (x - np.mean(x)) / np.std(x)
        acorr = np.array([
            _compute_auto_corr(x_std, lag) for lag in range(1, ar_order + 1)
        ])
        auto_corr_time = (1 - np.inner(acorr, ar_coef)) / (1 - np.sum(ar_coef)) ** 2

    ess = len(x) / auto_corr_time
    return ess

def batch_means(samples, axis=0, normed=False, n_batch=25):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the method of batch means.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    n_sample = samples.shape[axis]
    if 2 * n_batch > n_sample:
        raise ValueError(
            "The number of batches must be less than twice the number of samples."
        )

    batch_index = np.linspace(0, n_sample, n_batch + 1).astype('int')
    batch_list = [
        np.take(samples, np.arange(batch_index[i], batch_index[i + 1]), axis)
        for i in range(n_batch)
    ]
    batch_mean = np.stack((np.mean(batch, axis) for batch in batch_list), axis)
    mcmc_var = n_sample / n_batch * np.var(batch_mean, axis)
    ess = np.var(samples, axis) / mcmc_var
    if not normed: ess *= n_sample

    return ess

def monotone_sequence(
        samples, axis=0, normed=False, require_acorr=False):
    """
    Estimates effective sample sizes of samples along the specified axis
    with the monotone positive sequence estimator of "Practical Markov
    Chain Monte Carlo" by Geyer (1992). The estimator is ONLY VALID for
    reversible Markov chains. The inputs 'mu' and 'sigma_sq' are optional
    and unnecessary for the most cases in practice.

    Parameters
    ----------
    require_acorr : bool
        If true, a list of estimated auto correlation sequences are returned.

    Returns
    -------
    ess : numpy array
    auto_cor : list of numpy array
        auto-correlation estimates of the chain up to the lag beyond which the
        auto-correlation can be considered insignificant by the monotonicity
        criterion.
    """

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    n_param = samples.shape[1 - axis]
    n_sample = samples.shape[axis]
    ess = np.zeros(n_param)
    if n_sample <= 2: # Edge case.
        return ess

    auto_cor = []
    for j in range(n_param):
        if axis == 0:
            x = samples[:, j]
        else:
            x = samples[j, :]
        x_std = (x - np.mean(x)) / np.std(x)
        ess_j, auto_cor_j = _monotone_sequence_1d(x_std, require_acorr)
        ess[j] = ess_j
        if require_acorr:
            auto_cor.append(auto_cor_j)
    if normed:
        ess /= n_sample

    return ess, auto_cor

def _monotone_sequence_1d(x, require_acorr):
    """ The time series `x` is assumed to be standardized. """

    auto_corr = []

    # lag in [0, 1] case.
    lag_one_auto_corr = _compute_auto_corr(x, lag=1)
    running_min = 1. + lag_one_auto_corr
    auto_corr_sum = 1. + 2 * lag_one_auto_corr
    if require_acorr:
        auto_corr.extend((1., lag_one_auto_corr))
    curr_lag = 2

    while curr_lag + 2 < len(x):

        even_auto_corr, odd_auto_corr = [
            _compute_auto_corr(x, lag) for lag in [curr_lag, curr_lag + 1]
        ]
        curr_lag += 2
        if even_auto_corr + odd_auto_corr < 0:
            break

        running_min = min(running_min, (even_auto_corr + odd_auto_corr))
        auto_corr_sum += 2 * running_min
        if require_acorr:
            auto_corr.extend((even_auto_corr, odd_auto_corr))

    ess = len(x) / auto_corr_sum
    if auto_corr_sum < 0:
        # Rare, but can happen with floating point errors when the time series
        # `x` shows strong negative correlations.
        ess = float('inf')

    return ess, np.array(auto_corr)

def _compute_auto_corr(x, lag):
    """
    Returns an estimate of the lag 'k' auto-correlation of a time series 'x'.
    The estimator is biased towards zero due to the factor (len(x) - lag) / len(x).
    See Geyer (1992) Section 3.1 and the reference therein for justification.
    """
    acorr = np.mean(x[:-lag] * x[lag:]) * (len(x) - lag) / len(x)
    return acorr
