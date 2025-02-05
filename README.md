<div align=center>
    <img src="docs/source/static/logo.svg" alt="pyHaiCS Logo" height="200">
</div>

# pyHaiCS - Hamiltonian-based Monte-Carlo for Computational Statistics (HaiCS) in Python

## Implemented Samplers
The following samplers have been implemented in `pyHaiCS`:
### Monte Carlo Samplers (Non-Markov Chain)
- Direct Sampling
- Rejection Sampling
- Adaptive Rejection Sampling
- Importance Sampling
- Sampling Importance Re-sampling (SIR)
- Adaptive Importance Sampling (AIS)
- Inverse Transform Sampling
### Basic Markov Chain Monte Carlo Samplers
- (Random-Walk) Metropolis-Hastings Sampling
- Gibbs Sampling
- Metropolis-Adjusted Langevin Sampling
- Slice Sampling
- No-U-Turn Sampling
### Hamiltonian Markov Chain Monte Carlo Samplers
- Hamiltonian Monte Carlo (HMC) Sampling
- Generalized Hamiltonian Monte Carlo (GHMC) Sampling
- Molecular Dynamics Monte-Carlo (MDMC) Sampling
- Generalized Shadow Hamiltonian Monte Carlo (GSHMC) Sampling
- Mix & Match Hamiltonian Monte Carlo (MMHMC) Sampling

## General API Overview
- `Analysis`: Contains analytical tools for sampling analysis.
- `Config`: Contains configuration files for running `pyHaiCS`.
- `Samplers`: Contains all the implemented samplers (see section above).
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
