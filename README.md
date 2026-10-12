<div align=center>
    <img src="docs/source/static/logo.svg" alt="pyHaiCS Logo" height="200">
</div>

# pyHaiCS - Hamiltonian-based Monte-Carlo for Computational Statistics in Python

Introducing `pyHaiCS`, a Python library for Hamiltonian-based Monte-Carlo methods tailored towards practical applications in *computational statistics*. From sampling complex probability distributions, to approximating complex integrals — such as in the context of Bayesian inference — `pyHaiCS` is designed to be fast, flexible, and easy to use, with a focus on providing a user-friendly interface for researchers and practitioners while also offering users a variety of *advanced features*. 

Our library currently implements a wide range of *sampling algorithms* — including single-chain and multi-chain Hamiltoninan Monte-Carlo (HMC) and Generalized HMC (GHMC); a variety of numerical schemes for the *integration* of the simulated Hamiltonian dynamics (including a generalized version of Multi-Stage Splitting integrators), or a novel *adaptive* algorithm — Adaptive Integration Approach in Computational Statistics (ATune) — for the automatic tuning of the parameters of both the numerical integrator and the sampler. 

Likewise, several utilities for *diagnosing* the convergence and efficiency of the sampling process, as well as *multidisciplinary* benchmarks — ranging from simple toy problems such as sampling from specific distributions, to more complex real-world applications in the fields of computational biology, Bayesian modeling, or physics — are provided.

The main features of `pyHaiCS` include its:

- **Efficient Implementation:** `pyHaiCS` is built on top of the JAX library developed by Google, which provides *automatic differentiation* for computing gradients and Hessians, and Just-In-Time (JIT) *compilation* for fast numerical computations. Additionally, the library is designed to take advantage of multi-core CPUs, GPUs, or even TPUs for *accelerated* sampling, and to be highly *parallelizable* (e.g., by running each chain of multi-chain HMC in a separate CPU core/thread in the GPU).

- **User-Friendly Interface:** The library is designed to be easy to use, with a simple and intuitive API that abstracts away the complexities of Hamiltonian Monte-Carlo (HMC) and related algorithms. Users can define their own potential functions and priors, and run sampling algorithms with just a few lines of code.

- **Integration with Existing Tools:** The library is designed to be *easily integrated* with other Python libraries, such as `NumPy`, `SciPy`, and `Scikit-Learn`. This allows users to leverage existing tools and workflows, and build on top of the rich ecosystem of scientific computing in Python. Therefore, users can easily incorporate `pyHaiCS` into their existing Machine Learning workflows, and use it for tasks such as inference, model selection, or parameter estimation in the context of Bayesian modeling.

- **Advanced Features:** `pyHaiCS` supports a variety of Hamiltonian-inspired sampling algorithms, including single-chain and multi-chain HMC (and GHMC), generalized $k$-th stage Multi-Stage Splitting integrators, and adaptive integration schemes (such as ATune).

> [!NOTE] 
> `pyHaiCS` is an open-source project and is actively maintained and developed by a team of researchers and practitioners in the field of computational statistics. We welcome contributions from the community, and encourage users to report issues, suggest new features, and contribute to the development of the library ;)

> [!NOTE]
> This project was initially conceived as a Python implementation of the work presented in the following publications. However, the library has since evolved into a more general-purpose tool for Hamiltonian-based Monte-Carlo methods in computational statistics and now includes a variety of additional features and functionalities.
> - L. Nagar, M. Fernández-Pendás, J. M. Sanz-Serna, E. Akhmatskaya, Adaptive Multi-stage Integration Schemes for Hamiltonian Monte Carlo. Journal of Computational Physics, 502 (2024) 112800. DOI: https://doi.org/10.1016/j.jcp.2024.112800
> - T. Radivojevic, E. Akhmatskaya, Modified Hamiltonian Monte Carlo for Bayesian Inference. Statistics and Computing, 30 (2020) 377-404. DOI: https://doi.org/10.1007/s11222-019-09885-x 
> - T. Radivojevic. Enhancing Sampling in Computational Statistics Using Modified Hamiltonians. PhD thesis, UPV/EHU, Bilbao (Spain), 2016. URI http://hdl.handle.net/20.500.11824/323

## General Features of `pyHaiCS`
- `Samplers`: Contains Hamiltonian (and regular) MCMC samplers such as RWMH, HMC, GHMC, MHMC, RMHMC, Gibbs, Importance and Rejection sampling.
- `Integrators`: A variety of numerical integration for Hamiltonian dynamics are currently implemented: Leapfrog-Verlet, 2-stage & 3-stage MSSIs, Velocity-Verlet, BCSS, ME, etc.
- `Adaptive Tuning`: The _ATune_ sampler automatically tunes the step size, the number of integration steps and the momentum noise, and selects the optimal coefficients of the multi stage integrator, based on the s-AIA scheme.
- `Sampling Metrics`: A variety of metrics related to the quality of the sampling procedures.
- `Multi-Disciplinary Benchmarks`: Benchmarks provided to evaluate the samplers including applications in computational biology, Bayesian modeling, and physics, as well as toy problems for testing and validation.

## Installation
You can use `pip` to install `pyHaiCS` from the GitHub official release builds. You can do this by running the following command in your terminal:

```bash
pip install $(curl -s https://pyhaics.github.io/latest.txt)
```

Note that the above command will install the latest *official* version of `pyHaiCS` currently available on the GitHub releases page.

## General API Overview
- `pyHaiCS.samplers`: Contains all the implemented samplers, as top level functions
    - `hamiltonian`: `HMC`, `GHMC`, `MHMC`, `RMHMC`, `MALA`, `L2MC`, `MDMC`, `SLDMC`.
    - `adaptive`: `ATune` and `sMAIA` (MHMC counterpart of ATune)
    - `basic_mcmc`: `RWMH` and `gibbs_sampling`
    - `standard`: `rejection_sampling` and `importance_sampling`
- `pyHaiCS.integrators`: Contains all the integrators, as subclasses of `Integrator` class.
    - `VerletIntegrator`: Velocity Verlet, one stage.
    - `MSSI_2`: 2-stages Multi Stage Splitting Integrators (`VV_2`, `BCSS_2`, `ME_2`)
    - `MSSI_3`: 3-stages Multi Stage Splitting Integrators (`VV_3`, `BCSS_3`, `ME_3`)
    - `sAIA_2`, `sAIA_3`: adaptive MSSI integrators, with optimal choice of coefficients
    - `RMHMCVerletIntegrator`
- `pyHaiCS.utils`
    - `hamiltonian`: contains `MassMatrix`, `Hamiltonian`, `Modified Hamiltonian` and Fisher metrics.
    - `metrics`: contains acceptance rate, ESS, PSRF, MCSE, IACT and `compute_metrics` which prints a summary table.
    - `estimators` and `coda_utils`: estimators and helpers for the diagnostics and ESS metrics.
- `pyHaiCS.analysis`: reserved for analysis tools, currently empty
  
Other folders in the repository:
- `pyHaiCS/benchmarks`: datasets for the benchmark models (GD, BLR, BNN, PDE, epidemiological)
- `pyHaiCS/tests`: unit tests (for running all of them, use `./run_tests.sh`)

## Instructions to Run the Talbot Docker Environment

**To build the image:**
```bash
docker build --no-cache -t talbot .
```

**To create the container:**
```bash
docker run -it --name talbot -v ./:/talbot -w /talbot talbot
```

**To reconnect to the container:**
```bash
docker exec -it talbot /bin/bash
```

**To stop the container:**
```bash
docker stop talbot
```

**To remove the image:**
```bash
docker rmi talbot
```

**To remove the container:**
```bash
docker rm talbot
```

## TODO: Pending Tasks

* Improve Multi-Chain sampling for the adaptive samplers.
* Implement More Integrators
* Implement More Samplers
* Implement GRADe.
* Add tests for numerical correctness.
* Implement Adaptive Methods: ATune is implemented but limited to HMC/GHMC sampling and 2- & 3-stage splitting integrators
