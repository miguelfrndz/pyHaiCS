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
    samples = _as_3d_samples(samples)
    n_chains, n_samples, _ = samples.shape
    if n_chains < 2:
        return jnp.full((samples.shape[2],), jnp.nan)

    chain_means = np.mean(samples, axis = 1)
    within = np.mean(np.var(samples, axis = 1, ddof = 1), axis = 0)
    between = n_samples * np.var(chain_means, axis = 0, ddof = 1)
    variance_estimate = ((n_samples - 1) / n_samples) * within + between / n_samples
    rhat = np.sqrt(np.maximum(variance_estimate / np.maximum(within, 1e-12), 1.0))
    return jnp.asarray(rhat)


def _combine_with_weight_ess(ess, weights, normalize):
    if weights is None:
        return ess
    weight_ess = np.asarray(importance_sample_size(weights, normalize = False), dtype = float)
    combined = np.minimum(np.asarray(ess, dtype = float), weight_ess[:, None])
    if normalize:
        combined = combined / weights.shape[1]
    return combined


def _geyerESS_atomic(samples, thres_estimator):
    estimator = _GEYER_ALIASES.get(thres_estimator)
    if estimator is None:
        raise ValueError(f"Unknown threshold estimator: {thres_estimator}")

    rho = autocorrelation_sequence(samples, max_lag = len(samples) - 1)
    n_samples = len(samples)
    if rho[0] <= 0:
        return 1.0

    if estimator in ["initial_positive", "initial_monotone"]:
        pair_sums = []
        for k in range((len(rho) - 1) // 2):
            value = rho[2 * k] + rho[2 * k + 1]
            if value <= 0:
                break
            pair_sums.append(value)
        if estimator == "initial_monotone" and pair_sums:
            pair_sums = np.minimum.accumulate(pair_sums)
        tau = -1.0 + 2.0 * np.sum(pair_sums) if len(pair_sums) > 0 else 1.0
    elif estimator == "sign_trunc":
        positive = rho[1:]
        cutoff = np.argmax(positive < 0)
        if cutoff == 0 and positive[0] >= 0:
            cutoff = len(positive)
        tau = 1.0 + 2.0 * np.sum(positive[:cutoff])
    elif estimator == "lag_trunc":
        positive = rho[1:]
        cutoff = np.argmax(positive < 0.05)
        if cutoff == 0 and positive[0] >= 0.05:
            cutoff = len(positive)
        tau = 1.0 + 2.0 * np.sum(positive[:cutoff])
    elif estimator == "var_trunc":
        lags = np.arange(1, len(rho))
        tau = 1.0 + 2.0 * np.sum(rho[1:] * (1.0 - lags / n_samples))
    else:
        raise ValueError(f"Unsupported Geyer estimator: {estimator}")

    # Antithetic chains may legitimately have tau < 1 and therefore ESS > N.
    # We only guard against non-positive estimates caused by finite-sample noise.
    tau = max(tau, 1.0 / n_samples)
    ess = n_samples / tau
    return float(max(ess, 1.0))


def geyerESS(samples, thres_estimator = "IMSE", normalize = True, weights = None):
    samples = _as_3d_samples(samples)
    n_chains, n_samples, n_dims = samples.shape
    weights = _as_2d_weights(weights, n_chains, n_samples)

    ess_values = np.zeros((n_chains, n_dims), dtype = float)
    for chain in range(n_chains):
        for dim in range(n_dims):
            ess_values[chain, dim] = _geyerESS_atomic(samples[chain, :, dim], thres_estimator)

    ess_values = _combine_with_weight_ess(ess_values, weights, normalize = normalize)
    if weights is None and normalize:
        ess_values = ess_values / n_samples
    return jnp.asarray(ess_values)


def _batch_size_grid(n_samples, n_dims, batch_size, Nb):
    if isinstance(batch_size, str):
        if batch_size == "sqroot":
            return [int(np.floor(np.sqrt(n_samples)))]
        if batch_size == "cuberoot":
            return [int(np.floor(n_samples ** (1.0 / 3.0)))]
        if batch_size == "less":
            b_min = max(2, int(np.floor(n_samples ** 0.25)))
            b_max = max(int(np.floor(n_samples / max(n_dims, 20))), int(np.floor(np.sqrt(n_samples))))
            if Nb is None:
                Nb = 64
            grid = np.unique(np.round(np.exp(np.linspace(np.log(b_min), np.log(max(b_min + 1, b_max)), Nb)))).astype(int)
            return [int(b) for b in grid if 1 < b < n_samples / 2]
        raise ValueError("Unknown batch size string. Use 'sqroot', 'cuberoot', or 'less'.")

    if not 1 < batch_size < (n_samples / 2):
        raise ValueError("The batch size B needs to be between 1 and N/2.")
    return [int(batch_size)]


def _multiESS_batch(X, theta, det_lambda, b, Noffsets):
    n_samples, n_dims = X.shape
    a = n_samples // b
    if a <= 1:
        return np.nan

    max_offset = n_samples - a * b
    offsets = np.unique(np.round(np.linspace(0, max_offset, Noffsets)).astype(int))
    sigma_total = np.zeros((n_dims, n_dims), dtype = float)

    for offset in offsets:
        chunk = X[offset : offset + a * b]
        batch_means = chunk.reshape(a, b, n_dims).mean(axis = 1)
        centered = batch_means - theta
        sigma_total += centered.T @ centered

    sigma = (sigma_total * b) / ((a - 1) * len(offsets))
    det_sigma = float(np.linalg.det(sigma))
    if not np.isfinite(det_sigma) or det_sigma <= 0 or det_lambda <= 0:
        return np.nan
    return n_samples * (det_lambda / det_sigma) ** (1.0 / n_dims)


def _multiESS_chain(samples, batch_size = "sqroot", Noffsets = 10, Nb = None):
    X = np.asarray(samples, dtype = float)
    if X.ndim != 2:
        raise ValueError("A single chain must have shape (n_samples, n_dims)")
    n_samples, n_dims = X.shape
    if n_dims > n_samples:
        raise ValueError("More dimensions than data points; cannot compute multiESS.")

    theta = np.mean(X, axis = 0)
    covariance = np.atleast_2d(np.cov(X, rowvar = False))
    det_lambda = float(np.linalg.det(covariance))
    if not np.isfinite(det_lambda) or det_lambda <= 0:
        return np.nan

    batch_grid = _batch_size_grid(n_samples, n_dims, batch_size, Nb)
    estimates = [
        _multiESS_batch(X, theta, det_lambda, b, Noffsets)
        for b in batch_grid
    ]
    estimates = np.asarray(estimates, dtype = float)
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return np.nan
    return float(np.min(estimates))


def multiESS(samples, batch_size = "sqroot", Noffsets = 10, Nb = None, normalize = True, weights = None, combined = False):
    samples = _as_3d_samples(samples)
    n_chains, n_samples, _ = samples.shape
    weights = _as_2d_weights(weights, n_chains, n_samples)

    if combined:
        pooled = samples.reshape(-1, samples.shape[2])
        ess = _multiESS_chain(pooled, batch_size = batch_size, Noffsets = Noffsets, Nb = Nb)
        if weights is not None:
            pooled_weight_ess = float(importance_sample_size(weights.reshape(-1), normalize = False)[0])
            ess = min(ess, pooled_weight_ess)
        if normalize and np.isfinite(ess):
            ess /= pooled.shape[0]
        return jnp.asarray(ess)

    ess_values = np.asarray([
        _multiESS_chain(samples[chain], batch_size = batch_size, Noffsets = Noffsets, Nb = Nb)
        for chain in range(n_chains)
    ], dtype = float)

    if weights is not None:
        weight_ess = np.asarray(importance_sample_size(weights, normalize = False), dtype = float)
        ess_values = np.minimum(ess_values, weight_ess)
    if normalize:
        ess_values = ess_values / n_samples
    return jnp.asarray(ess_values)


def codaESS(samples, axis = 0, method = "ar", normed = False, options = None, normalize = True, weights = None):
    samples = _as_3d_samples(samples)
    n_chains, n_samples, n_dims = samples.shape
    weights = _as_2d_weights(weights, n_chains, n_samples)
    if options is None:
        options = {}
    if n_samples <= 25:
        warnings.warn(
            "The number of samples is extremely small. The estimated ESS will likely be unreliable."
        )

    ess_values = np.zeros((n_chains, n_dims), dtype = float)
    for chain in range(n_chains):
        if method == "ar":
            max_ar_order = options.get("max_ar_order", None)
            ess_chain = ar_process_fit(samples[chain], axis, normed = False, max_ar_order = max_ar_order)
        elif method == "monotone-sequence":
            ess_chain, _ = monotone_sequence(samples[chain], axis, normed = False)
        elif method == "batch-means":
            n_batch = options.get("n_batch", 25)
            ess_chain = batch_means(samples[chain], axis, normed = False, n_batch = n_batch)
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
    step_grid = np.asarray(step_grid, dtype = float)
    ess_values = np.asarray(ess_values, dtype = float)
    if normalized_ESS:
        return jnp.asarray(1.0 / np.maximum(ess_values, 1e-12))
    return jnp.asarray(step_grid / np.maximum(ess_values, 1e-12))


def _summary(values):
    values = np.asarray(values, dtype = float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.min(values)), float(np.mean(values)), float(np.max(values)))


def compute_metrics(samples, thres_estimator = "IMSE", normalize_ESS = True, weights = None, pacf_lags = (1, 5, 10), print_results = True):
    samples = _as_3d_samples(samples)
    weights = _as_2d_weights(weights, samples.shape[0], samples.shape[1])

    diagnostics = {
        "psrf": PSRF(samples),
        "geyer_ess": geyerESS(samples, thres_estimator = thres_estimator, normalize = normalize_ESS, weights = weights),
        "coda_ess": codaESS(samples, method = "monotone-sequence", normalize = normalize_ESS, weights = weights),
        "multi_ess": multiESS(samples, normalize = normalize_ESS, weights = weights),
    }

    raw_ess = geyerESS(samples, thres_estimator = thres_estimator, normalize = False, weights = weights)
    diagnostics["mcse"] = MCSE(samples, raw_ess, weights = weights)
    diagnostics["iact"] = IACT(samples, raw_ess, normalized_ESS = False)

    max_pacf_lag = max(pacf_lags) if pacf_lags else 0
    pacf_values = PACF(samples, max_lag = max_pacf_lag)
    diagnostics["pacf"] = {lag: jnp.abs(pacf_values[:, :, lag]) for lag in pacf_lags}

    diagnostics["summary"] = {
        "psrf": _summary(diagnostics["psrf"]),
        "geyer_ess": _summary(diagnostics["geyer_ess"]),
        "coda_ess": _summary(diagnostics["coda_ess"]),
        "multi_ess": _summary(diagnostics["multi_ess"]),
        "mcse": _summary(diagnostics["mcse"]),
        "iact": _summary(diagnostics["iact"]),
        "pacf": {lag: _summary(values) for lag, values in diagnostics["pacf"].items()},
    }

    if print_results:
        print("Sampling Results:")
        print(f"{' ':<10} {'PSRF':<10} {'ESS':<10} {'CODA':<10} {'mESS':<10} {'MCSE':<10} {'IACT':<10}")
        s = diagnostics["summary"]
        print(f"{'Min':<10} {s['psrf'][0]:<10.2f} {s['geyer_ess'][0]:<10.2f} {s['coda_ess'][0]:<10.2f} {s['multi_ess'][0]:<10.2f} {s['mcse'][0]:<10.2f} {s['iact'][0]:<10.2f}")
        print(f"{'Avg':<10} {s['psrf'][1]:<10.2f} {s['geyer_ess'][1]:<10.2f} {s['coda_ess'][1]:<10.2f} {s['multi_ess'][1]:<10.2f} {s['mcse'][1]:<10.2f} {s['iact'][1]:<10.2f}")
        print(f"{'Max':<10} {s['psrf'][2]:<10.2f} {s['geyer_ess'][2]:<10.2f} {s['coda_ess'][2]:<10.2f} {s['multi_ess'][2]:<10.2f} {s['mcse'][2]:<10.2f} {s['iact'][2]:<10.2f}\n")
        if pacf_lags:
            headers = " ".join([f"PACF@{lag}".rjust(12) for lag in pacf_lags])
            print(f"{' ':<10} {headers}")
            print(f"{'Avg':<10} " + " ".join([f"{s['pacf'][lag][1]:>12.4f}" for lag in pacf_lags]))
            print()

    return diagnostics
