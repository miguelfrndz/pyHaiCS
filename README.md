<div align=center>
    <img src="docs/source/static/logo.svg" alt="pyHaiCS Logo" style="height: auto; max-width: 55%;">
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