import argparse
import time

import jax
import jax.numpy as jnp
import pyHaiCS as haics

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@jax.jit
def model_fn(x, params):
    return jax.nn.sigmoid(jnp.matmul(x, params))


@jax.jit
def log_prior_fn(params):
    return jnp.sum(jax.scipy.stats.norm.logpdf(params))


@jax.jit
def log_likelihood_fn(x, y, params):
    epsilon = 1e-7
    preds = model_fn(x, params)
    return jnp.sum(y * jnp.log(preds + epsilon) + (1 - y) * jnp.log(1 - preds + epsilon))


@jax.jit
def neg_log_posterior_fn(x, y, params):
    return -(log_prior_fn(params) + log_likelihood_fn(x, y, params))


def build_breast_cancer_problem(seed):
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = seed, stratify = y
    )

    scaler = StandardScaler()
    X_train = jnp.array(scaler.fit_transform(X_train))
    X_test = jnp.array(scaler.transform(X_test))

    X_train = jnp.hstack([X_train, jnp.ones((X_train.shape[0], 1))])
    X_test = jnp.hstack([X_test, jnp.ones((X_test.shape[0], 1))])

    key = jax.random.PRNGKey(seed)
    mean_vector = jnp.zeros(X_train.shape[1])
    cov_mat = jnp.eye(X_train.shape[1])
    params = jax.random.multivariate_normal(key, mean_vector, cov_mat)

    return params, X_train, y_train, X_test, y_test


def run_independent_chains(single_chain_fn, n_chains, base_seed):
    chains = []
    for chain_idx in range(n_chains):
        chains.append(single_chain_fn(base_seed + chain_idx))
    return jnp.stack(chains, axis = 0)


def ensure_chain_shape(samples):
    if samples.ndim == 2:
        samples = samples[None, :, :]
    return samples


def summarize_array(values):
    values = jnp.ravel(values)
    values = values[jnp.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(jnp.min(values)), float(jnp.mean(values)), float(jnp.max(values))


def safe_multi_ess_summary(samples):
    try:
        pooled = haics.utils.metrics.multiESS(samples, normalize = True, combined = True)
        value = float(jnp.asarray(pooled))
        return value, value, value
    except ValueError:
        return float("nan"), float("nan"), float("nan")


def pacf_summary(samples, lags):
    pacf_values = jnp.abs(haics.utils.metrics.PACF(samples, max_lag = max(lags)))
    summary = {}
    for lag in lags:
        summary[lag] = summarize_array(pacf_values[:, :, lag])
    return summary


def predictive_accuracy(samples, X_test, y_test):
    if samples.ndim == 3:
        samples = jnp.mean(samples, axis = 0)
    preds = jax.vmap(lambda params: model_fn(X_test, params))(samples)
    mean_preds = jnp.mean(preds, axis = 0) > 0.5
    return float(jnp.mean(mean_preds == y_test))


def run_and_summarize(name, fn, estimator, pacf_lags, X_test, y_test):
    start = time.perf_counter()
    samples = fn()
    runtime = time.perf_counter() - start
    samples = ensure_chain_shape(samples)
    ess_raw = haics.utils.metrics.geyerESS(samples, thres_estimator = estimator, normalize = False)
    ess_norm = haics.utils.metrics.geyerESS(samples, thres_estimator = estimator, normalize = True)
    coda_ess_norm = haics.utils.metrics.codaESS(samples, method = "monotone-sequence", normalize = True)
    multi_ess_norm = safe_multi_ess_summary(samples)
    psrf = summarize_array(haics.utils.metrics.PSRF(samples))
    mcse = haics.utils.metrics.MCSE(samples, ess_raw)
    iact = haics.utils.metrics.IACT(samples, ess_raw, normalized_ESS = False)
    pacf = pacf_summary(samples, pacf_lags)
    min_ess, mean_ess, max_ess = summarize_array(ess_norm)
    accuracy = predictive_accuracy(samples, X_test, y_test)
    return {
        "name": name,
        "min_ess": min_ess,
        "mean_ess": mean_ess,
        "max_ess": max_ess,
        "coda_ess": summarize_array(coda_ess_norm),
        "multi_ess": multi_ess_norm,
        "psrf": psrf,
        "mcse": summarize_array(mcse),
        "iact": summarize_array(iact),
        "pacf": pacf,
        "accuracy": accuracy,
        "runtime": runtime,
    }


def main():
    parser = argparse.ArgumentParser(
        description = "Compare normalized ESS for HMC, GHMC and s-AIA on Bayesian logistic regression."
    )
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--n-chains", type = int, default = 4)
    parser.add_argument("--stage", type = int, default = 3, choices = [2, 3])
    parser.add_argument("--ess-estimator", type = str, default = "IMSE")
    parser.add_argument("--n-samples", type = int, default = 1000)
    parser.add_argument("--burn-in", type = int, default = 200)
    parser.add_argument("--hmc-step-size", type = float, default = 1e-3)
    parser.add_argument("--hmc-n-steps", type = int, default = 100)
    parser.add_argument("--ghmc-step-size", type = float, default = 1e-3)
    parser.add_argument("--ghmc-n-steps", type = int, default = 100)
    parser.add_argument("--ghmc-momentum-noise", type = float, default = 0.3)
    parser.add_argument("--saia-tune", type = int, default = 400)
    parser.add_argument("--saia-check", type = int, default = 100)
    parser.add_argument("--saia-burn-in", type = int, default = 600)
    parser.add_argument("--saia-prod", type = int, default = 1000)
    parser.add_argument("--pacf-lags", type = str, default = "1,5,10")
    args = parser.parse_args()
    pacf_lags = [int(lag.strip()) for lag in args.pacf_lags.split(",") if lag.strip()]

    params, X_train, y_train, X_test, y_test = build_breast_cancer_problem(args.seed)
    mass_matrix = jnp.eye(X_train.shape[1])

    runs = [
        (
            "HMC",
            lambda: haics.samplers.hamiltonian.HMC(
                params,
                potential_args = (X_train, y_train),
                n_samples = args.n_samples,
                burn_in = args.burn_in,
                step_size = args.hmc_step_size,
                n_steps = args.hmc_n_steps,
                potential = neg_log_posterior_fn,
                mass_matrix = mass_matrix,
                integrator = haics.integrators.VerletIntegrator(),
                n_chains = args.n_chains,
                RNG_key = args.seed,
            ),
        ),
        (
            "GHMC",
            lambda: haics.samplers.hamiltonian.GHMC(
                params,
                potential_args = (X_train, y_train),
                n_samples = args.n_samples,
                burn_in = args.burn_in,
                step_size = args.ghmc_step_size,
                n_steps = args.ghmc_n_steps,
                potential = neg_log_posterior_fn,
                mass_matrix = mass_matrix,
                momentum_noise = args.ghmc_momentum_noise,
                integrator = haics.integrators.VerletIntegrator(),
                n_chains = args.n_chains,
                RNG_key = args.seed,
            ),
        ),
        (
            "s-AIA (HMC)",
            lambda: run_independent_chains(
                lambda seed: haics.samplers.adaptive.sAIA(
                    params,
                    potential_args = (X_train, y_train),
                    n_samples_tune = args.saia_tune,
                    n_samples_check = args.saia_check,
                    n_samples_burn_in = args.saia_burn_in,
                    n_samples_prod = args.saia_prod,
                    potential = neg_log_posterior_fn,
                    mass_matrix = mass_matrix,
                    target_AR = 0.92,
                    stage = args.stage,
                    sensibility = 0.01,
                    delta_step = 0.01,
                    compute_freqs = True,
                    compute_hessian = True,
                    sampler = "HMC",
                    RNG_key = seed,
                ),
                n_chains = args.n_chains,
                base_seed = args.seed + 1_000,
            ),
        ),
        (
            "s-AIA (GHMC)",
            lambda: run_independent_chains(
                lambda seed: haics.samplers.adaptive.sAIA(
                    params,
                    potential_args = (X_train, y_train),
                    n_samples_tune = args.saia_tune,
                    n_samples_check = args.saia_check,
                    n_samples_burn_in = args.saia_burn_in,
                    n_samples_prod = args.saia_prod,
                    potential = neg_log_posterior_fn,
                    mass_matrix = mass_matrix,
                    target_AR = 0.92,
                    stage = args.stage,
                    sensibility = 0.01,
                    delta_step = 0.01,
                    compute_freqs = True,
                    compute_hessian = True,
                    sampler = "GHMC",
                    RNG_key = seed,
                ),
                n_chains = args.n_chains,
                base_seed = args.seed + 2_000,
            ),
        ),
    ]

    results = [
        run_and_summarize(name, fn, args.ess_estimator, pacf_lags, X_test, y_test)
        for name, fn in runs
    ]

    print(f"{'Sampler':<14} {'minESS/N':>12} {'meanESS/N':>12} {'maxESS/N':>12} {'Accuracy':>10} {'Time(s)':>10}")
    print("-" * 74)
    for result in results:
        print(
            f"{result['name']:<14} "
            f"{result['min_ess']:>12.4f} "
            f"{result['mean_ess']:>12.4f} "
            f"{result['max_ess']:>12.4f} "
            f"{result['accuracy']:>10.4f} "
            f"{result['runtime']:>10.2f}"
        )

    print()
    print(f"{'Sampler':<14} {'mean CODA':>12} {'pooled multiESS':>16} {'mean PSRF':>12} {'mean MCSE':>12} {'mean IACT':>12}")
    print("-" * 92)
    for result in results:
        print(
            f"{result['name']:<14} "
            f"{result['coda_ess'][1]:>12.4f} "
            f"{result['multi_ess'][1]:>14.4f} "
            f"{result['psrf'][1]:>12.4f} "
            f"{result['mcse'][1]:>12.4f} "
            f"{result['iact'][1]:>12.4f}"
        )

    print()
    pacf_headers = " ".join([f"PACF@{lag:>2}".rjust(12) for lag in pacf_lags])
    print(f"{'Sampler':<14} {pacf_headers}")
    print("-" * (16 + 13 * len(pacf_lags)))
    for result in results:
        pacf_means = " ".join([f"{result['pacf'][lag][1]:>12.4f}" for lag in pacf_lags])
        print(f"{result['name']:<14} {pacf_means}")


if __name__ == "__main__":
    main()
