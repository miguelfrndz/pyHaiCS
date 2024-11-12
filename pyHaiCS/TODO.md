# TODO: Pending Tasks

* Improve Multi-Chain sampling (better parallelization)
* Make integrators JIT-Compilable (Maybe using PyTrees or Partially-Static Compiling) -> As of right now, the code for the Verlet integrator does actually JIT-Compile, but it can be severely improved
* Implement More Integrators
* Implement More Samplers
* Implement Adaptive Methods: s-AIA is implemented but limited to HMC sampling and 2- & 3-stage splitting integrators
* A performance issue related to s-AIA needs fixing: because in the production stage each iteration of HMC uses a different L, epsilon, and Phi (integrator), the pre-compiled JIT versions are not as efficient (massive compilation overhead each time).
* Implement Geyer's Effective Sample Size using JAX and VMap so that the computation can be vectorized accross chains and paramters of the model.