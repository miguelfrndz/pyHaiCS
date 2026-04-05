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


def normalized_ess_summary(samples, estimator):
    if samples.ndim == 2:
        samples = samples[None, :, :]
    ess = haics.utils.metrics.geyerESS(samples, thres_estimator = estimator, normalize = True)
    ess = jnp.ravel(ess)
    return float(jnp.min(ess)), float(jnp.mean(ess)), float(jnp.max(ess))


def predictive_accuracy(samples, X_test, y_test):
    if samples.ndim == 3:
        samples = jnp.mean(samples, axis = 0)
    preds = jax.vmap(lambda params: model_fn(X_test, params))(samples)
    mean_preds = jnp.mean(preds, axis = 0) > 0.5
    return float(jnp.mean(mean_preds == y_test))


def run_and_summarize(name, fn, estimator, X_test, y_test):
    start = time.perf_counter()
    samples = fn()
    runtime = time.perf_counter() - start
    min_ess, mean_ess, max_ess = normalized_ess_summary(samples, estimator)
    accuracy = predictive_accuracy(samples, X_test, y_test)
    return {
        "name": name,
        "min_ess": min_ess,
        "mean_ess": mean_ess,
        "max_ess": max_ess,
        "accuracy": accuracy,
        "runtime": runtime,
    }


def main():
    parser = argparse.ArgumentParser(
        description = "Compare normalized ESS for HMC, GHMC and s-AIA on Bayesian logistic regression."
    )
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--stage", type = int, default = 3, choices = [2, 3])
    parser.add_argument("--ess-estimator", type = str, default = "var_trunc")
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
    args = parser.parse_args()

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
                n_chains = 1,
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
                n_chains = 1,
                RNG_key = args.seed,
            ),
        ),
        (
            "s-AIA (HMC)",
            lambda: haics.samplers.adaptive.sAIA(
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
                RNG_key = args.seed,
            ),
        ),
        (
            "s-AIA (GHMC)",
            lambda: haics.samplers.adaptive.sAIA(
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
                RNG_key = args.seed,
            ),
        ),
    ]

    results = [
        run_and_summarize(name, fn, args.ess_estimator, X_test, y_test)
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


if __name__ == "__main__":
    main()
