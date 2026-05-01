"""
s-AIA sampling example for the Gaussian Distribution benchmark.

The GD benchmark files contain:
    - D[dim]_div.txt: precision matrix, i.e. inverse covariance.
    - D[dim]_div_eig.txt: sorted precision eigenvalues, i.e. squared frequencies.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyHaiCS as haics


@jax.jit
def gaussian_potential(precision_matrix, params):
    return 0.5 * params @ (precision_matrix @ params)


def benchmark_paths(dim):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "pyHaiCS" / "benchmarks" / "GD"
    return data_dir / f"D{dim}_div.txt", data_dir / f"D{dim}_div_eig.txt"


def load_gd_benchmark(dim):
    precision_path, eig_path = benchmark_paths(dim)
    if not precision_path.exists() or not eig_path.exists():
        raise FileNotFoundError(
            f"Missing GD benchmark files for D={dim}: {precision_path.name}, {eig_path.name}"
        )

    precision_matrix = np.loadtxt(precision_path, dtype = np.float32)
    precision_eigs = np.loadtxt(eig_path, dtype = np.float32)
    return jnp.asarray(precision_matrix), precision_matrix, precision_eigs


def run_independent_chains(args, precision_matrix, mass_matrix):
    chains = []
    x_init = jnp.zeros(args.dim)
    for chain_idx in range(args.n_chains):
        samples = haics.samplers.adaptive.sAIA(
            x_init,
            potential_args = (precision_matrix,),
            n_samples_tune = args.n_samples_tune,
            n_samples_check = args.n_samples_check,
            n_samples_burn_in = args.n_samples_burn_in,
            n_samples_prod = args.n_samples_prod,
            potential = gaussian_potential,
            mass_matrix = mass_matrix,
            target_AR = args.target_ar,
            stage = args.stage,
            sensibility = args.sensibility,
            delta_step = args.delta_step,
            compute_freqs = args.compute_hessian,
            compute_hessian = args.compute_hessian,
            sampler = args.sampler,
            RNG_key = args.seed + chain_idx,
        )
        chains.append(samples)
    return jnp.stack(chains, axis = 0)


def selected_true_variances(precision_matrix_np, selected_dims):
    rhs = np.eye(precision_matrix_np.shape[0], dtype = precision_matrix_np.dtype)[:, selected_dims]
    covariance_columns = np.linalg.solve(precision_matrix_np, rhs)
    return covariance_columns[selected_dims, np.arange(len(selected_dims))]


def summarize_samples(samples, precision_matrix_np, precision_eigs, diagnostic_dims):
    selected_dims = np.arange(min(diagnostic_dims, samples.shape[-1]))
    flat_samples = samples.reshape(-1, samples.shape[-1])
    selected_samples = np.asarray(flat_samples[:, selected_dims])

    sample_means = np.mean(selected_samples, axis = 0)
    sample_vars = np.var(selected_samples, axis = 0)
    true_vars = selected_true_variances(precision_matrix_np, selected_dims)
    rel_var_error = np.abs(sample_vars - true_vars) / np.maximum(true_vars, 1e-12)

    print()
    print("Benchmark summary")
    print(f"  Dimension: {samples.shape[-1]}")
    print(f"  Precision eigenvalue range: [{precision_eigs[0]:.6g}, {precision_eigs[-1]:.6g}]")
    print(f"  Frequency range: [{np.sqrt(precision_eigs[0]):.6g}, {np.sqrt(precision_eigs[-1]):.6g}]")
    print()
    print(f"Diagnostics on first {len(selected_dims)} coordinates")
    print(f"  RMS sample mean: {np.sqrt(np.mean(sample_means ** 2)):.6g}")
    print(f"  Mean relative variance error: {np.mean(rel_var_error):.6g}")
    print(f"  Max relative variance error: {np.max(rel_var_error):.6g}")

    geyer_ess = haics.utils.metrics.geyerESS(samples, thres_estimator = "IMSE", normalize = True)
    geyer_ess_selected = np.asarray(geyer_ess)[:, selected_dims]
    print(f"  Normalized Geyer IMSE ESS, selected dims: min={np.min(geyer_ess_selected):.4f}, mean={np.mean(geyer_ess_selected):.4f}, max={np.max(geyer_ess_selected):.4f}")

    coda_ess = haics.utils.metrics.codaESS(samples, method = "monotone-sequence", normalize = True)
    coda_ess_selected = np.asarray(coda_ess)[:, selected_dims]
    print(f"  Normalized CODA monotone ESS, selected dims: min={np.min(coda_ess_selected):.4f}, mean={np.mean(coda_ess_selected):.4f}, max={np.max(coda_ess_selected):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description = "Sample the GD multivariate Gaussian benchmark with s-AIA."
    )
    parser.add_argument("--dim", type = int, default = 1000)
    parser.add_argument("--sampler", choices = ["HMC", "GHMC"], default = "GHMC")
    parser.add_argument(
        "--stage",
        type = int,
        choices = [2, 3],
        default = 3,
        help = "Integrator stage. Stage 2 is the default for the precision-preconditioned GD Gaussian.",
    )
    parser.add_argument(
        "--mass-matrix",
        choices = ["precision", "identity"],
        default = "precision",
        help = "Use precision preconditioning for this Gaussian, or identity for the unpreconditioned benchmark.",
    )
    parser.add_argument("--n-chains", type = int, default = 1)
    parser.add_argument("--n-samples-tune", type = int, default = 2000)
    parser.add_argument("--n-samples-check", type = int, default = 2000)
    parser.add_argument("--n-samples-burn-in", type = int, default = 2500)
    parser.add_argument("--n-samples-prod", type = int, default = 2500)
    parser.add_argument("--target-ar", type = float, default = 0.92)
    parser.add_argument("--sensibility", type = float, default = 0.01)
    parser.add_argument("--delta-step", type = float, default = 0.001)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--diagnostic-dims", type = int, default = parser.get_default("dim"), help = "Number of leading dimensions to include in diagnostics.")
    parser.add_argument(
        "--compute-hessian",
        action = "store_true",
        dest = "compute_hessian",
        help = "Force Hessian-based frequency estimation.",
    )
    parser.add_argument(
        "--no-compute-hessian",
        action = "store_false",
        dest = "compute_hessian",
        help = "Use the cheaper no-frequency s-AIA branch instead of Hessian-based frequencies.",
    )
    parser.set_defaults(compute_hessian = None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compute_hessian is None:
        args.compute_hessian = args.mass_matrix == "identity"

    print(f"Running pyHaiCS v.{haics.__version__}")
    print(f"Loading GD benchmark D={args.dim}...")
    precision_matrix, precision_matrix_np, precision_eigs = load_gd_benchmark(args.dim)
    if args.mass_matrix == "precision":
        mass_matrix = precision_matrix
    else:
        mass_matrix = jnp.eye(args.dim, dtype = precision_matrix.dtype)
    print(f"Mass matrix: {args.mass_matrix}")
    print(f"Compute Hessian frequencies: {'yes' if args.compute_hessian else 'no'}")

    start = time.perf_counter()
    samples = run_independent_chains(args, precision_matrix, mass_matrix)
    runtime = time.perf_counter() - start

    print()
    print(f"s-AIA {args.sampler} finished in {runtime:.2f} seconds")
    print(f"Samples shape: {samples.shape}")
    summarize_samples(samples, precision_matrix_np, precision_eigs, args.diagnostic_dims)


if __name__ == "__main__":
    main()
