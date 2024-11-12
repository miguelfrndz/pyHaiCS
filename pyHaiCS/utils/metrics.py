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

def compute_metrics(samples, thres_estimator = 'IMSE', normalize_ESS = True):
    """
    Compute the PSRF, ESS, MCSE, and IACT values for a given set of samples.
    Prints the results in a tabular format (min/avg/max values).

    Parameters:
    -----------
    samples: jnp.ndarray
        The samples from the MCMC chains. Shape: (n_chains, n_samples, n_dims)
    thres_estimator: str
        The threshold estimator to use for the ESS calculation. Default: 'IMSE'
    normalize_ESS: bool
        Whether to normalize the ESS values. Default: True
    """
    # Compute the PSRF, ESS, MCSE, and IACT values
    psrf_values = PSRF(samples)
    # Average across chains + Reshape to consider as one chain
    samples = jnp.mean(samples, axis = 0)
    samples = jnp.reshape(samples, (1, samples.shape[0], samples.shape[1]))
    ess_values = geyerESS(samples, thres_estimator, normalize = normalize_ESS)
    mcse_values = MCSE(samples, ess_values)
    iact_values = IACT(samples, ess_values, normalized_ESS = normalize_ESS)
    # print(f"Potential Scale Reduction Factor (PSRF): {psrf_values}\n")
    # print(f"Effective Sample Size (ESS-Geyer-IMSE): {ess_values}\n")
    # print(f"Monte Carlo Standard Error (MCSE): {mcse_values}\n")
    # print(f"Integrated Autocorrelation Time (IACT): {iact_values}\n")

    # Compute the min/avg/max of the ess/mcse/iact values
    minPSRF, avgPSRF, maxPSRF = jnp.min(psrf_values), jnp.mean(psrf_values), jnp.max(psrf_values)
    minESS, avgESS, maxESS = jnp.min(ess_values), jnp.mean(ess_values), jnp.max(ess_values)
    minMCSE, avgMCSE, maxMCSE = jnp.min(mcse_values), jnp.mean(mcse_values), jnp.max(mcse_values)
    minIACT, avgIACT, maxIACT = jnp.min(iact_values), jnp.mean(iact_values), jnp.max(iact_values)

    # Print results table
    print("Sampling Results:")
    print(f"{' ':<10} {'PSRF':<10} {'ESS':<10} {'MCSE':<10} {'IACT':<10}")
    print(f"{'Min':<10} {minPSRF:<10.2f} {minESS:<10.2f} {minMCSE:<10.2f} {minIACT:<10.2f}")
    print(f"{'Avg':<10} {avgPSRF:<10.2f} {avgESS:<10.2f} {avgMCSE:<10.2f} {avgIACT:<10.2f}")
    print(f"{'Max':<10} {maxPSRF:<10.2f} {maxESS:<10.2f} {maxMCSE:<10.2f} {maxIACT:<10.2f}\n")