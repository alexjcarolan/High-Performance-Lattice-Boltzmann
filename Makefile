# Makefile

EXE = lbm

OMP = lbm_omp.c
OCL = lbm_ocl.c
MPI = lbm_mpi.c

OMP-CC = icc
OCL-CC = icc
MPI-CC = mpiicc

OMP-CFLAGS = -std=c99 -Wall -Ofast -ipo -qopenmp -mtune=native -march=native
OCL-CFLAGS = -std=c99 -Wall -Ofast -ipo -qopenmp -mtune=native -march=native
MPI-CFLAGS = -std=c99 -Wall -Ofast -ipo -qopenmp -mtune=native -march=native

OMP-LFLAGS = -ipo -lm -o
OCL-LFLAGS = -lOpenCL -ipo -lm -o
MPI-LFLAGS = -ipo -lm -o

.PHONY: omp ocl mpi clean

omp: $(OMP)
	$(OMP-CC) $(OMP-CFLAGS) $(OMP) $(OMP-LFLAGS) $(EXE)

ocl: $(OCL)
	$(OCL-CC) $(OCL-CFLAGS) $(OCL) $(OCL-LFLAGS) $(EXE)

mpi: $(MPI)
	$(MPI-CC) $(MPI-CFLAGS) $(MPI) $(MPI-LFLAGS) $(EXE)

clean:
	rm -f $(EXE)