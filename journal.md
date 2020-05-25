# Weekly Progress Journal

### Week 1 - Planning and Project Proposal
Goal: Simulate the dynamics of Carbon-13 around NV centres in diamond,
and investigate the coherent control of individual Carbons and pairs. 

We wish to solve the Lindblad equation for the evolution of Carbon-13 near NV centres in
diamond. This solver will constitute the main computational task of the project, which we
will then apply to finding interesting results about the system in the form of plots that
can be compared to recent literature in the field.

We want to start by finding how a dynamical decoupling sequence of varying tau value (the 
free evolution parameter) affects the state of the electron in the NV. (When the electron 
interacts with a Carbon atom, it results in a noticeable change in the electron's state since
the two spins become entangled). 

We have done some research and found a few keys resources:
1. Recent papers outlining the decoupling sequence and how the electron state can be 
computed from the physical parameters of the system (this information makes the first plot
we want to create less of a computationally difficult task and more an exercise in plotting
a long function).
2. A Python library called QuTiP. This is a very valuable quantum toolbox, including such 
things as a master equation solver for time evolution of a system directly from a given 
Hamiltonian, plotting on the Bloch sphere and other cool visualization help, etc. Some of
the functions in this package we will use to compare to our master equation solver.
