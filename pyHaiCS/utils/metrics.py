import jax
import warnings
import jax.numpy as jnp
import numpy as np

from .coda_utils import ar_process_fit, monotone_sequence, batch_means

def autocovariance(samples, lag = 1):
    """
    Calculates the autocovariance of a series of samples.
    Note that at lag = 0, the autocovariance is equal to the variance of the samples.
    """
    n_samples = len(samples)
    mean = np.mean(samples)
    return np.mean((samples[:n_samples - lag] - mean) * (samples[lag:] - mean))

def acceptance_rate(num_acceptals, n_samples):
    return num_acceptals / n_samples

def rejection_rate(num_acceptals, n_samples):
    return 1 - acceptance_rate(num_acceptals, n_samples)

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

def _geyerESS_atomic(samples, thres_estimator, normalize):
    if thres_estimator not in ['ISPE', 'IMSE', 'var_trunc', 'lag_trunc', 'sign_trunc']:
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

def geyerESS(samples, thres_estimator = 'var_trunc', normalize = True):
    """
    Calculates the Geyer's Effective Sample Size (ESS) of a series of samples.
    """
    # TODO: Currently uses Numpy instead of JAX. Implement JAX version with vectorized operations for chains/dimensions.
    samples = np.array(samples)
    n_chains, n_samples, dims = samples.shape
    ess_values = []
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
            "The number of samples is extremely small. The estimated ESS "
            "will likely be not reliable at all."
        )
    ess_values = []
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
            raise NotImplementedError("Method {:s} is not supported.".format(method))
        if normalize:
            ess = ess / n_samples
        ess_values.append(ess)
    return jnp.array(ess_values).reshape(n_chains, dims)


def MCSE(samples, ess_values):
    """
    Monte-Carlo Standard Error (MCSE).
    """
    return jnp.std(samples, axis = 1) / jnp.sqrt(ess_values)

def IACT(samples, ess_values, normalized_ESS = True):
    """
    Integrated Autocorrelation Time (IACT). The number of Monte-Carlo iterations needed, on average, 
    for an independent sample to be drawn.
    """
    if normalized_ESS:
        return samples.shape[1] / (ess_values * samples.shape[1])
    return samples.shape[1] / ess_values

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