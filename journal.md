# Weekly Progress Journal

## Planning and Project Proposal (due May 27)
Goal: Simulate the dynamics of Carbon-13 around NV centres in diamond,
and investigate the coherent control of individual Carbons and pairs. 

We wish to solve the Lindblad equation for the evolution of Carbon-13 near NV centres in
diamond. This solver will constitute the main computational task of the project, which we
will then apply to finding interesting results about systems in the form of plots that
can be compared to recent literature in the field.

We want to start by finding how a dynamical decoupling sequence of varying tau value (the 
free evolution parameter) affects the state of the electron in the NV. (When the electron 
interacts with a Carbon atom, it results in a noticeable change in the electron's state since
the two spins become entangled). 

We have done some research and found a few keys resources:
1. Recent papers outlining the decoupling sequence and how the electron state can be 
computed from the physical parameters of the system.
2. A Python library called QuTiP. This is a very valuable quantum toolbox, including such 
things as a master equation solver for time evolution of a system directly from a given 
Hamiltonian, plotting on the Bloch sphere and other cool visualization help, etc. Some of
the functions in this package we will use to compare to our master equation solver.

## Milestones
### Week 1 (due May 27th)
- [X] Complete a provisional master equation solver
- [ ] Implement tox

- [X] Started creating examples of dynamical systems using the [lindblad solver]
      function, which can be found in the
      [examples](https://gitlab.kwant-project.org/computational_physics_projects/Project-3_A.Galicia_nicholaszutt/-/tree/master/examples)
      folder.
      
      1. We verified that the Lindbladian was implementing the right dynamics by showing the precession of a spin under the influence of a magnetic field in the z direction. We can show how tilting the angle of precession

### Week 2 (due June 3rd)
