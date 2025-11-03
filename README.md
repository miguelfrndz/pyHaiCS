<div align=center>
    <img src="docs/source/static/logo.svg" alt="pyHaiCS Logo" height="200">
</div>

# pyHaiCS - Hamiltonian-based Monte-Carlo for Computational Statistics in Python

Introducing `pyHaiCS`, a Python library for Hamiltonian-based Monte-Carlo methods tailored towards practical applications in *computational statistics*. From sampling complex probability distributions, to approximating complex integrals — such as in the context of Bayesian inference — `pyHaiCS` is designed to be fast, flexible, and easy to use, with a focus on providing a user-friendly interface for researchers and practitioners while also offering users a variety of *advanced features*. 

Our library currently implements a wide range of *sampling algorithms* — including single-chain and multi-chain Hamiltoninan Monte-Carlo (HMC) and Generalized HMC (GHMC); a variety of numerical schemes for the *integration* of the simulated Hamiltonian dynamics (including a generalized version of Multi-Stage Splitting integrators), or a novel *adaptive* algorithm — Adaptive Integration Approach in Computational Statistics (s-AIA) — for the automatic tuning of the parameters of both the numerical integrator and the sampler. 

Likewise, several utilities for *diagnosing* the convergence and efficiency of the sampling process, as well as *multidisciplinary* benchmarks — ranging from simple toy problems such as sampling from specific distributions, to more complex real-world applications in the fields of computational biology, Bayesian modeling, or physics — are provided.

The main features of `pyHaiCS` include its:

- **Efficient Implementation:** `pyHaiCS` is built on top of the JAX library developed by Google, which provides *automatic differentiation* for computing gradients and Hessians, and Just-In-Time (JIT) *compilation* for fast numerical computations. Additionally, the library is designed to take advantage of multi-core CPUs, GPUs, or even TPUs for *accelerated* sampling, and to be highly *parallelizable* (e.g., by running each chain of multi-chain HMC in a separate CPU core/thread in the GPU).

- **User-Friendly Interface:** The library is designed to be easy to use, with a simple and intuitive API that abstracts away the complexities of Hamiltonian Monte-Carlo (HMC) and related algorithms. Users can define their own potential functions and priors, and run sampling algorithms with just a few lines of code.

- **Integration with Existing Tools:** The library is designed to be *easily integrated* with other Python libraries, such as `NumPy`, `SciPy`, and `Scikit-Learn`. This allows users to leverage existing tools and workflows, and build on top of the rich ecosystem of scientific computing in Python. Therefore, users can easily incorporate `pyHaiCS` into their existing Machine Learning workflows, and use it for tasks such as inference, model selection, or parameter estimation in the context of Bayesian modeling.

- **Advanced Features:** `pyHaiCS` supports a variety of Hamiltonian-inspired sampling algorithms, including single-chain and multi-chain HMC (and GHMC), generalized $k$-th stage Multi-Stage Splitting integrators, and adaptive integration schemes (such as s-AIA).

> [!NOTE] 
> `pyHaiCS` is an open-source project and is actively maintained and developed by a team of researchers and practitioners in the field of computational statistics. We welcome contributions from the community, and encourage users to report issues, suggest new features, and contribute to the development of the library ;)

> [!NOTE]
> This project was initially conceived as a Python implementation of the work presented in the following publications:
> - L. Nagar, M. Fernández-Pendás, J. M. Sanz-Serna, E. Akhmatskaya, Adaptive Multi-stage Integration Schemes for Hamiltonian Monte Carlo. Journal of Computational Physics, 502 (2024) 112800. DOI: https://doi.org/10.1016/j.jcp.2024.112800
> - T. Radivojevic, E. Akhmatskaya, Modified Hamiltonian Monte Carlo for Bayesian Inference. Statistics and Computing, 30 (2020) 377-404. DOI: https://doi.org/10.1007/s11222-019-09885-x 
> - T. Radivojevic. Enhancing Sampling in Computational Statistics Using Modified Hamiltonians. PhD thesis, UPV/EHU, Bilbao (Spain), 2016. URI http://hdl.handle.net/20.500.11824/323
> However, the library has since evolved into a more general-purpose tool for Hamiltonian-based Monte-Carlo methods in computational statistics and now includes a variety of additional features and functionalities.

## General Features of `pyHaiCS`
- `Samplers`: Contains Hamiltonian (and regular) MCMC samplers such as RWMH, HMC, GHMC.
- `Integrators`: A variety of numerical integration for Hamiltonian dynamics are currently implemented: Leapfrog-Verlet, 2-stage & 3-stage MSSIs, Velocity-Verlet, BCSS, ME, etc.
- `Adaptive Tuning`: The s-AIA tuning scheme is implemented for automatically estimating the best integrator and sampler parameters.
- `Sampling Metrics`: A variety of metrics related to the quality of the sampling procedures.
- `Multi-Disciplinary Benchmarks`: Benchmarks provided to evaluate the samplers including applications in computational biology, Bayesian modeling, and physics, as well as toy problems for testing and validation.

## Installation
You can use `pip` to install `pyHaiCS` from the GitHub official release builds. You can do this by running the following command in your terminal:

```bash
pip install $(curl -s https://pyhaics.github.io/latest.txt)
```

Note that the above command will install the latest *official* version of `pyHaiCS` currently available on the GitHub releases page.

## General API Overview (TO BE UPDATED)
- `Analysis`: Contains analytical tools for sampling analysis.
- `Config`: Contains configuration files for running `pyHaiCS`.
- `Samplers`: Contains all the implemented samplers (see section below).
- `Tests`: Test folder. Preferably, run the `run_tests.sh` instead of running each test individually. You might be required to give execution rights to the script using the following command: `chmod +x run_tests.sh`.
- `Utils`: Contains general utilities used by `pyHaiCS`.

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

* Improve Multi-Chain sampling (better parallelization)
* Make integrators JIT-Compilable (Maybe using PyTrees or Partially-Static Compiling) -> As of right now, the code for the Verlet integrator does actually JIT-Compile, but it can be severely improved
* Implement More Integrators
* Implement More Samplers
* Implement Adaptive Methods: s-AIA is implemented but limited to HMC/GHMC sampling and 2- & 3-stage splitting integrators
* A performance issue related to s-AIA needs fixing: because in the production stage each iteration of HMC uses a different L, epsilon, and Phi (integrator), the pre-compiled JIT versions are not as efficient (massive compilation overhead each time).
* Implement Geyer's Effective Sample Size using JAX and VMap so that the computation can be vectorized accross chains and paramters of the model.
