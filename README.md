# pyHaiCS
```
    ┓┏  •┏┓┏┓
┏┓┓┏┣┫┏┓┓┃ ┗┓
┣┛┗┫┛┗┗┻┗┗┛┗┛
┛  ┛         
```
Python Library for Hamiltonian Markov Chain Monte Carlo in Computational Statistics (HaiCS).

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
- Metropolis-Hastings Sampling
- Gibbs Sampling
- Metropolis-Adjusted Langevin Sampling
- Slice Sampling
- No-U-Turn Sampling
### Hamiltonian Markov Chain Monte Carlo Samplers
- Hamiltonian Monte Carlo (HMC) Sampling
- Generalized Hamiltonian Monte Carlo (GHMC) Sampling
- Generalized Shadow Hamiltonian Monte Carlo (GSHMC) Sampling
- Mix & Match Hamiltonian Monte Carlo (MMHMC) Sampling

## General API Overview
- `Analysis`: Contains analytical tools for sampling analysis.
- `Config`: Contains configuration files for running `pyHaiCS`.
- `Samplers`: Contains all the implemented samplers (see section above).
- `Tests`: Test folder. Preferably, run the `run_tests.sh` instead of running each test individually. You might be required to give execution rights to the script using the following command: `chmod +x run_tests.sh`.
- `Utils`: Contains general utilities used by `pyHaiCS`.
