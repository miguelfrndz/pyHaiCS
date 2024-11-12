import jax
import jax.numpy as jnp
import numpy as np

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
    within_chain_variance = jnp.mean(jnp.var(samples, axis = 1), axis = 0)
    global_param_means = jnp.mean(jnp.mean(samples, axis = 1), axis = 0)
    between_chain_variance = n_samples/(n_chains - 1) * jnp.sum((jnp.mean(samples, axis = 1) - global_param_means) ** 2, axis = 0)
    sample_variance_estimate = (1 - 1/n_samples) * within_chain_variance + between_chain_variance/n_samples
    variance_estimate = sample_variance_estimate + between_chain_variance/(n_samples * n_chains)
    degs_of_freedom = 2 * variance_estimate ** 2 / jnp.var(variance_estimate)
    return jnp.sqrt((degs_of_freedom + 3) / (degs_of_freedom + 1) * variance_estimate / within_chain_variance)

def _geyerESS_atomic(samples, thres_estimator, normalize):
    if thres_estimator not in ['ISPE', 'IMSE', 'var_trunc', 'lag_trunc', 'sign_trunc']:
        raise ValueError(f"Unknown threshold estimator: {thres_estimator}")
    n_samples = samples.shape[0]
    if thres_estimator in ['ISPE', 'IMSE']:
        pairwise_autocovs = []
        sum_pairwise_autocovs = 0
        for k in range(n_samples):
            pairwise_autocov = autocovariance(samples, 2*k) + autocovariance(samples, 2*k + 1)
            if pairwise_autocov < 0:
                break
            if thres_estimator == 'IMSE':
                # Pairwise autocovariance is kept monotonically decreasing
                if k > 0 and pairwise_autocov > pairwise_autocovs[-1]:
                    pairwise_autocov = pairwise_autocovs[-1]
            sum_pairwise_autocovs += pairwise_autocov
            pairwise_autocovs.append(pairwise_autocov)
        ESS = n_samples / (-1 + 2 * sum_pairwise_autocovs)
    elif thres_estimator in ['lag_trunc', 'sign_trunc']:
        sum_autocovs = 0
        for k in range(1, 2*n_samples + 1):
            autocov = autocovariance(samples, k)
            if thres_estimator == 'lag_trunc' and autocov < 0.05:
                break
            elif thres_estimator == 'sign_trunc' and autocov < 0:
                break
            sum_autocovs += autocov
        ESS = n_samples / (1 + 2 * sum_autocovs)
    elif thres_estimator == 'var_trunc':
        sum_autocovs = 0
        for k in range(n_samples):
            autocov = autocovariance(samples, k)
            sum_autocovs += autocov * (n_samples - k) / n_samples
        ESS = n_samples / (1 + 2 * sum_autocovs)
    if normalize:
        return ESS / n_samples
    return ESS

def geyerESS(samples, thres_estimator = 'IMSE', normalize = True):
    """
    Calculates the Geyer's Effective Sample Size (ESS) of a series of samples.
    """
    # TODO: Currently uses Numpy instead of JAX. Implement JAX version with vectorized operations for chains/dimensions.
    samples = np.array(samples)
    print("Samples Shape for Geyer ESS:", samples.shape)
    n_chains, n_samples, dims = samples.shape
    ess_values = []
    for chain in range(n_chains):
        for dim in range(dims):
            samples_iter = samples[chain, :, dim]
            ess_value = _geyerESS_atomic(samples_iter, thres_estimator, normalize)
            ess_values.append(ess_value)
    return jnp.array(ess_values).reshape(n_chains, dims)

def MCSE(samples, ess_values):
    """
    Monte-Carlo Standard Error (MCSE).
    """
    return jnp.std(samples, axis = 1) / jnp.sqrt(ess_values)

def IACT(samples, ess_values):
    """
    Integrated Autocorrelation Time (IACT). The number of Monte-Carlo iterations needed, on average, 
    for an independent sample to be drawn.
    """
    return samples.shape[1] / ess_values
    