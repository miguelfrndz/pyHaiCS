# TODO: Pending Tasks

* Improve Multi-Chain sampling (better parallelization)
* Make integrators JIT-Compilable (Maybe using PyTrees or Partially-Static Compiling) -> As of right now, the code for the Verlet integrator does actually JIT-Compile, but it can be severely improved
* Implement More Integrators
* Implement More Samplers
* Implement Adaptive Methods: s-AIA is implemented but limited to HMC sampling and 2- & 3-stage splitting integrators