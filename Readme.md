# High Performance Lattice Boltzmann
A high performance lattice Boltzmann fluid simulator operating in 2 dimensions with 9 speeds, optimised for both shared memory & distributed memory platforms as well as homogeneous & heterogeneous platforms. Consisting of 3 separate implementations which together comprise the major parallel programming paradigms of OMP, OCL & MPI. Further detail can be found in the following reports [1](Report_1.pdf) [2](Report_2.pdf).

|Simulation|
|----------|
|![](outputs/simulation.gif)|

|OMP Scaling|MPI Scaling|
|-----------|-----------|
|![](outputs/omp_scaling.png)|![](outputs/mpi_scaling.png)|

|OMP Roofline|OCL Roofline|MPI Roofline|
|------------|------------|------------|
|![](outputs/omp_roofline.png)|![](outputs/ocl_roofline.png)|![](outputs/mpi_roofline.png)|