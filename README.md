# High Performance Lattice Boltzmann
A high performance lattice Boltzmann fluid simulator operating in 2 dimensions with 9 speeds, optimised for both shared memory & distributed memory platforms as well as homogeneous & heterogeneous platforms. Consisting of 3 separate implementations which together comprise the major parallel programming paradigms of OMP, OCL & MPI. Further detail can be found in the following reports [1](report_1.pdf) [2](report_2.pdf).

|Simulation|
|----------|
|![](outputs/simulation.gif)|

|OMP Scaling|OMP Roofline|
|-----------|-----------|
|![](outputs/omp_scaling.png)|![](outputs/omp_roofline.png)|

|OCL Roofline|
|------------|
|![](outputs/ocl_roofline.png)|

|MPI Scaling|MPI Roofline|
|-----------|-----------|
|![](outputs/mpi_scaling.png)|![](outputs/mpi_roofline.png)|