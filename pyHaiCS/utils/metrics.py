import warnings

import jax.numpy as jnp
import numpy as np

from .coda_utils import ar_process_fit, monotone_sequence, batch_means


_GEYER_ALIASES = {
    "ISPE": "initial_positive",
    "IMSE": "initial_monotone",
    "initial_positive": "initial_positive",
    "initial_monotone": "initial_monotone",
    "var_trunc": "var_trunc",
    "lag_trunc": "lag_trunc",
    "sign_trunc": "sign_trunc",
}


def _as_3d_samples(samples):
    samples = np.asarray(samples, dtype = float)
    if samples.ndim == 1:
        samples = samples[None, :, None]
    elif samples.ndim == 2:
        samples = samples[None, :, :]
    elif samples.ndim != 3:
        raise ValueError("samples must have shape (n_chains, n_samples, n_dims)")
    return samples


def _as_2d_weights(weights, n_chains, n_samples):
    if weights is None:
        return None
    weights = np.asarray(weights, dtype = float)
    if weights.ndim == 1:
        if weights.shape[0] != n_samples:
            raise ValueError(f"weights must have length {n_samples}")
        weights = np.repeat(weights[None, :], n_chains, axis = 0)
    elif weights.ndim != 2:
        raise ValueError("weights must have shape (n_samples,) or (n_chains, n_samples)")
    if weights.shape != (n_chains, n_samples):
        raise ValueError(f"weights must have shape {(n_chains, n_samples)}")
    return weights


def _normalize_weights_1d(weights):
    weights = np.asarray(weights, dtype = float)
    weights = np.clip(weights, a_min = 0.0, a_max = None)
    total = np.sum(weights)
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def weighted_mean(samples, weights = None):
    samples = _as_3d_samples(samples)
    n_chains, n_samples, _ = samples.shape
    weights = _as_2d_weights(weights, n_chains, n_samples)
    if weights is None:
        return jnp.asarray(np.mean(samples, axis = 1))

    means = []
    for chain in range(n_chains):
        w = _normalize_weights_1d(weights[chain])
        means.append(np.sum(samples[chain] * w[:, None], axis = 0))
    return jnp.asarray(np.stack(means, axis = 0))


def weighted_variance(samples, weights = None, ddof = 0):
    samples = _as_3d_samples(samples)
    n_chains, n_samples, _ = samples.shape
    weights = _as_2d_weights(weights, n_chains, n_samples)
    if weights is None:
        return jnp.asarray(np.var(samples, axis = 1, ddof = ddof))

    variances = []
    for chain in range(n_chains):
        w = _normalize_weights_1d(weights[chain])
        mean = np.sum(samples[chain] * w[:, None], axis = 0)
        centered = samples[chain] - mean
        var = np.sum(w[:, None] * centered ** 2, axis = 0)
        if ddof != 0:
            denom = max(1.0 - np.sum(w ** 2), 1e-12)
            var = var / denom
        variances.append(var)
    return jnp.asarray(np.stack(variances, axis = 0))


def autocovariance(samples, lag = 1):
    samples = np.asarray(samples, dtype = float).reshape(-1)
    n_samples = samples.shape[0]
    if lag < 0 or lag >= n_samples:
        raise ValueError("lag must satisfy 0 <= lag < n_samples")
    centered = samples - np.mean(samples)
    return np.dot(centered[: n_samples - lag], centered[lag:]) / n_samples


def autocovariance_sequence(samples, max_lag = None):
    samples = np.asarray(samples, dtype = float).reshape(-1)
    n_samples = samples.shape[0]
    if max_lag is None:
        max_lag = n_samples - 1
    max_lag = min(max_lag, n_samples - 1)
    centered = samples - np.mean(samples)
    return np.asarray([
        np.dot(centered[: n_samples - lag], centered[lag:]) / n_samples
        for lag in range(max_lag + 1)
    ])


def autocorrelation_sequence(samples, max_lag = None):
    autocovs = autocovariance_sequence(samples, max_lag = max_lag)
    if autocovs[0] <= 0:
        return np.zeros_like(autocovs)
    return autocovs / autocovs[0]


def ACF(samples, max_lag = 20):
    samples = _as_3d_samples(samples)
    n_chains, _, n_dims = samples.shape
    acf_values = np.zeros((n_chains, n_dims, max_lag + 1), dtype = float)
    for chain in range(n_chains):
        for dim in range(n_dims):
            acf_values[chain, dim] = autocorrelation_sequence(samples[chain, :, dim], max_lag = max_lag)
    return jnp.asarray(acf_values)


def _pacf_1d(samples, max_lag):
    gamma = autocovariance_sequence(samples, max_lag = max_lag)
    pacf = np.zeros(max_lag + 1, dtype = float)
    pacf[0] = 1.0
    if gamma[0] <= 0:
        return pacf

    phi = np.zeros((max_lag + 1, max_lag + 1), dtype = float)
    innovation_var = gamma[0]
    for k in range(1, max_lag + 1):
        if k == 1:
            phi[k, k] = gamma[1] / gamma[0]
        else:
            numerator = gamma[k] - np.dot(phi[k - 1, 1:k], gamma[1:k][::-1])
            phi[k, k] = numerator / innovation_var
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        innovation_var *= (1.0 - phi[k, k] ** 2)
        pacf[k] = phi[k, k]
        if innovation_var <= 1e-12:
            break
    return pacf


def PACF(samples, max_lag = 20):
    samples = _as_3d_samples(samples)
    n_chains, _, n_dims = samples.shape
    pacf_values = np.zeros((n_chains, n_dims, max_lag + 1), dtype = float)
    for chain in range(n_chains):
        for dim in range(n_dims):
            pacf_values[chain, dim] = _pacf_1d(samples[chain, :, dim], max_lag = max_lag)
    return jnp.asarray(pacf_values)


def acceptance_rate(num_acceptals, n_samples):
    return num_acceptals / n_samples


def rejection_rate(num_acceptals, n_samples):
    return 1 - acceptance_rate(num_acceptals, n_samples)


def importance_sample_size(weights, normalize = True):
    weights = np.asarray(weights, dtype = float)
    if weights.ndim == 1:
        weights = weights[None, :]

    ess_values = []
    for chain_weights in weights:
        normalized = _normalize_weights_1d(chain_weights)
        ess = 1.0 / np.sum(normalized ** 2)
        if normalize:
            ess /= chain_weights.shape[0]
        ess_values.append(ess)
    return jnp.asarray(ess_values)


def PSRF(samples):
    """
    Gelman-Rubin Potential Scale Reduction Factor (PSRF) is a metric to assess convergence of MCMC chains.
    It is calculated as the square root of the ratio of the estimated marginal posterior variance of the parameters
    and the average of the estimated posterior variance within each chain.
    """
    n_chains, n_samples, dims = samples.shape
    if n_chains < 2:
        raise ValueError("PSRF requires at least 2 chains.")
    chain_means = jnp.mean(samples, axis = 1)
    within_chain_variance = jnp.mean(jnp.var(samples, axis = 1, ddof = 1), axis = 0)
    between_chain_variance = n_samples * jnp.var(chain_means, axis = 0, ddof = 1)
    variance_estimate = (n_samples - 1) / n_samples * within_chain_variance + between_chain_variance / n_samples
    return jnp.sqrt(variance_estimate / within_chain_variance)

def _geyerESS_atomic(samples, thres_estimator):
    estimator = _GEYER_ALIASES.get(thres_estimator)
    if estimator is None:
        raise ValueError(f"Unknown threshold estimator: {thres_estimator}")
    n_samples = samples.shape[0]
    var = autocovariance(samples, 0)
    if thres_estimator in ['ISPE', 'IMSE']:
        pairwise_autocorrs = []
        sum_pairwise_autocorrs = 0
        for k in range(n_samples//2):
            pairwise_autocorr = (autocovariance(samples, 2*k) + autocovariance(samples, 2*k + 1)) / var
            if pairwise_autocorr < 0:
                break
            if thres_estimator == 'IMSE':
                # Pairwise autocovariance is kept monotonically decreasing
                if k > 0 and pairwise_autocorr > pairwise_autocorrs[-1]:
                    pairwise_autocorr = pairwise_autocorrs[-1]
            sum_pairwise_autocorrs += pairwise_autocorr
            pairwise_autocorrs.append(pairwise_autocorr)
        ESS = n_samples / (-1 + 2 * sum_pairwise_autocorrs)
    elif thres_estimator in ['lag_trunc', 'sign_trunc']:
        sum_autocorrs = 0
        for k in range(1, n_samples):
            autocorr = autocovariance(samples, k) / var
            if thres_estimator == 'lag_trunc' and autocorr < 0.05:
                break
            elif thres_estimator == 'sign_trunc' and autocorr < 0:
                break
            sum_autocorrs += autocorr
        ESS = n_samples / (1 + 2 * sum_autocorrs)
    elif thres_estimator == 'var_trunc':
        max_lag = int(np.sqrt(n_samples))
        sum_autocorrs = 0
        for k in range(1, max_lag):
            autocorr = autocovariance(samples, k) / var
            sum_autocorrs += autocorr * (n_samples - k) / n_samples * (1 - k / max_lag)
        ESS = n_samples / (1 + 2 * sum_autocorrs)
    if normalize:
        return ESS / n_samples
    return ESS

def geyerESS(samples, thres_estimator = "IMSE", normalize = True, weights = None):
    samples = _as_3d_samples(samples)
    n_chains, n_samples, n_dims = samples.shape
    weights = _as_2d_weights(weights, n_chains, n_samples)

    ess_values = np.zeros((n_chains, n_dims), dtype = float)
    for chain in range(n_chains):
        for dim in range(dims):
            samples_iter = samples[chain, :, dim]
            ess_value = _geyerESS_atomic(samples_iter, thres_estimator, normalize)
            ess_values.append(ess_value)
    return jnp.array(ess_values).reshape(n_chains, dims)

def multiESS(samples, batch_size = 'sqroot', Noffsets = 10, Nb = None, normalize = True):
    """
    Adapted from: https://github.com/Gabriel-p/multiESS (no longer maintained)

    Compute multivariate effective sample size of a single Markov chain X,
    using the multivariate dependence structure of the process.

    X: MCMC samples of shape (n, p)
    n: number of samples
    p: number of parameters

    b: specifies the batch size for estimation of the covariance matrix in
       Markov chain CLT. It can take a numeric value between 1 and n/2, or a
       char value between:

    'sqroot'    b=floor(n^(1/2)) (for chains with slow mixing time; default)
    'cuberoot'  b=floor(n^(1/3)) (for chains with fast mixing time)
    'less'      pick the b that produces the lowest effective sample size
                for a number of b ranging from n^(1/4) to n/max(20,p); this
                is a conservative choice

    If n is not divisible by b Sigma is recomputed for up to Noffsets subsets
    of the data with different offsets, and the output mESS is the average over
    the effective sample sizes obtained for different offsets.

    Nb specifies the number of values of b to test when b='less'
    (default NB=200). This option is unused for other choices of b.

    If normalize is True, the effective sample size is normalized by the number of samples.

    Original source: https://github.com/lacerbi/multiESS

    Reference:
    Vats, D., Flegal, J. M., & Jones, G. L. "Multivariate Output Analysis
    for Markov chain Monte Carlo", arXiv preprint arXiv:1512.07713 (2015).

    """
    n_chains, n_samples, dims = samples.shape
    if dims > n_samples:
        raise ValueError("More dimensions than data points; cannot compute effective sample size.")
    if isinstance(batch_size, str):
        if batch_size not in ['sqroot', 'cuberoot', 'less']:
            raise ValueError("Unknown batch size string. Use 'sqroot', 'cuberoot', or 'less'.")
        if batch_size != 'less' and Nb is not None:
            print("Warning: Nb is ignored unless b='less'")
    else:
        if not 1. < batch_size < (n_samples / 2):
            raise ValueError("The batch size B needs to be between 1 and N/2.")
    ess_values = []
    for chain in range(n_chains):
        mESS = _multiESS_chain(samples[chain], n_samples, dims, batch_size, Noffsets, Nb)
        if normalize:
            mESS /= n_samples
        ess_values.append(mESS)
    return jnp.array(ess_values)

def _multiESS_chain(Xi, n, p, b, Noffsets, Nb):
    """
    Compute multiESS for a MCMC chain.
    """
    Xi = np.asarray(Xi, dtype = np.float64)
    if b == 'sqroot':
        b = [int(np.floor(n ** (1. / 2)))]
    elif b == 'cuberoot':
        b = [int(np.floor(n ** (1. / 3)))]
    elif b == 'less':
        b_min = np.floor(n ** (1. / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 200
        b = np.unique(np.round(np.exp(np.linspace(np.log(b_min), np.log(b_max), Nb)))).astype(int)
    else:
        b = [int(b)]

    theta = np.mean(Xi, axis = 0)
    _, logdetLambda = np.linalg.slogdet(np.atleast_2d(np.cov(Xi.T)))

    return min(_multiESS_batch(Xi, n, p, theta, logdetLambda, bi, Noffsets) for bi in b)

def _multiESS_batch(Xi, n, p, theta, logdetLambda, b, Noffsets):
    """
    Compute multiESS for a given batch size B.
    """
    a = int(np.floor(n / b))
    max_offset = n - a * b
    offsets = np.unique(np.round(np.linspace(0, max_offset, Noffsets)).astype(int))

    Sigma_total = np.zeros((p, p))
    for j in offsets:
        Z = Xi[j:j + a * b].reshape(a, b, p).mean(axis = 1) - theta
        Sigma_total += Z.T @ Z
    Sigma = Sigma_total * b / (a - 1) / len(offsets)

    _, logdetSigma = np.linalg.slogdet(Sigma)
    return n * np.exp((logdetLambda - logdetSigma) / p)

def codaESS(samples, axis = 0, method = 'ar', options = {}, normalize = True):
    """
    Wrapper for calling a function specified by the `method` argument.
    Estimates effective sample sizes (ESS) of samples along the specified axis.

    Parameters
    ----------
    samples : numpy 1d or 2d array
    axis : int, optional
    method : str, {'ar', 'monotone-sequence', 'batch-means'}
    options : dict with keys in ['max_ar_order', 'n_batch']
    normalize : bool

    Returns
    -------
    ess : numpy array
    """
    # TODO: Currently uses Numpy instead of JAX. Implement JAX version with vectorized operations for chains/dimensions.
    samples = np.array(samples)
    n_chains, n_samples, dims = samples.shape
    if n_samples <= 25:
        warnings.warn(
            "The number of samples is extremely small. The estimated ESS will likely be unreliable."
        )

    ess_values = np.zeros((n_chains, n_dims), dtype = float)
    for chain in range(n_chains):
        if method == 'ar':
            max_ar_order = options.get('max_ar_order', None)
            ess = ar_process_fit(samples[chain], axis, False, max_ar_order=max_ar_order)

        elif method == 'monotone-sequence':
            ess, _ = monotone_sequence(samples[chain], axis, False)

        elif method == 'batch-means':
            n_batch = options.get('n_batch', 25)
            ess = batch_means(samples[chain], axis, False, n_batch=n_batch)
        else:
            raise NotImplementedError(f"Method {method} is not supported.")
        ess_values[chain] = np.asarray(ess_chain, dtype = float)

    ess_values = _combine_with_weight_ess(ess_values, weights, normalize = False)
    if normalize or normed:
        ess_values = ess_values / n_samples
    return jnp.asarray(ess_values)


def MCSE(samples, ess_values, weights = None):
    samples = _as_3d_samples(samples)
    ess_values = np.asarray(ess_values, dtype = float)
    variances = np.asarray(weighted_variance(samples, weights = weights), dtype = float)
    mcse = np.sqrt(variances / np.maximum(ess_values, 1e-12))
    return jnp.asarray(mcse)


def IACT(samples, ess_values, normalized_ESS = False):
    samples = _as_3d_samples(samples)
    ess_values = np.asarray(ess_values, dtype = float)
    if normalized_ESS:
        return jnp.asarray(1.0 / np.maximum(ess_values, 1e-12))
    return jnp.asarray(samples.shape[1] / np.maximum(ess_values, 1e-12))


def GRADe(step_grid, ess_values, normalized_ESS = True):
    # TODO: Implement this as the ratio of the number of grad. computations and the ESS
    pass

def _print_metrics_table(metrics):
    print("Sampling Results:")
    print(f"{' ':<10} {'PSRF':<10} {'ESS':<10} {'MCSE':<10} {'IACT':<10}")
    for label, fn in [('Min', jnp.min), ('Avg', jnp.mean), ('Max', jnp.max)]:
        row = [fn(metrics[k]) for k in ('PSRF', 'ESS', 'MCSE', 'IACT')]
        print(f"{label:<10} " + " ".join(f"{v:<10.2f}" for v in row))
    print()

def compute_metrics(samples, thres_estimator = 'IMSE', normalize_ESS = True, verbose = True):
    """
    Compute the PSRF, ESS, MCSE, and IACT values for a given set of samples.
    Prints the results in a tabular format (min/avg/max values) if verbose is True.

    Parameters:
    -----------
    samples: jnp.ndarray
        The samples from the MCMC chains. Shape: (n_chains, n_samples, n_dims)
    thres_estimator: str
        The threshold estimator to use for the ESS calculation. Default: 'IMSE'
    normalize_ESS: bool
        Whether to report the ESS divided by the total number of samples. Default: True
    verbose: bool
        Whether to print the summary table. Default: True

    Returns:
    --------
    dict with keys 'PSRF', 'ESS', 'MCSE', 'IACT', each an array of shape (n_dims,)
    """
    psrf_values = PSRF(samples)
    n_chains, n_samples, n_dims = samples.shape
    # Absolute ESS per chain, summed over chains (assumes the chains have converged)
    ess_chains = geyerESS(samples, thres_estimator, normalize = False)
    ess_total = jnp.sum(ess_chains, axis = 0, keepdims = True)
    # Pool all chains into one long sample for std and IACT
    pooled = jnp.reshape(samples, (1, n_chains * n_samples, n_dims))
    mcse_values = MCSE(pooled, ess_total)[0]
    iact_values = IACT(pooled, ess_total, normalized_ESS = False)[0]
    ess_values = ess_total[0] / (n_chains * n_samples) if normalize_ESS else ess_total[0]

    metrics = {'PSRF': psrf_values, 'ESS': ess_values, 'MCSE': mcse_values, 'IACT': iact_values}
    if verbose:
        _print_metrics_table(metrics)
    return metrics