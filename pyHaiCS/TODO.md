# TODO: Pending Tasks

* Improve Multi-Chain sampling (better parallelization)
* Make integrators JIT-Compilable (Maybe using PyTrees or Partially-Static Compiling) -> As of right now, the code for the Verlet integrator does actually JIT-Compile. However, it is slower than the non-compiled version. 
* Implement More Integrators
* Implement More Samplers
* Implement Adaptive Methods