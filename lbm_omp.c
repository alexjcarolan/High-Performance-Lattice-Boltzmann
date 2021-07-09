// Lattice boltzmann d2q9-bgk scheme 
// 'd2'  indicates a 2-dimensional grid
// 'q9'  indicates 9 velocities per grid cell
// 'bgk' indicates the Bhatnagar-Gross-Krook step

// A 2D grid unwrapped row major to give a 1D array as follows:
//
//          ny  cols(ii)
//          ^   --- ---  
//          |  | C | D |     --- --- --- ---
// rows(jj) |   --- ---     | A | B | C | D |
//          |  | A | B |     --- --- --- ---
//          |   --- ---
//           ── ── ── ── > nx

// The speeds in each cell are ordered as follows:
//
//    6 2 5
//     ╲|╱
// 3 ── 0 ── 1
//     ╱|╲
//    7 4 8

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS  9
#define THREADS  28
#define BINDING  close
#define SCHEDULE static
#define FSFILE   "outputs/final_state.data"
#define AVFILE   "outputs/average_velocity.data"

// structure to hold parameters
typedef struct
{
  int   nx, ny;       // cells in each direction
  int   reynolds;     // reynolds dimensions
  int   iterations;   // number of iterations
  float density;      // density constant
  float relaxation;   // relaxation constant
  float acceleration; // acceleration consant
} parameters;

// structure to hold speeds
typedef struct
{
  float* s0; // speed in 0-direction 
  float* s1; // speed in 1-direction
  float* s2; // speed in 2-direction
  float* s3; // speed in 3-direction
  float* s4; // speed in 4-direction
  float* s5; // speed in 5-direction
  float* s6; // speed in 6-direction
  float* s7; // speed in 7-direction
  float* s8; // speed in 8-direction
} grid;

// function prototypes
float realise(const parameters parameters, grid* restrict old_grid, grid* restrict new_grid, const char* restrict obstacles, const int total_cells);
int initialise(parameters* parameters_ptr, grid** old_grid_ptr, grid** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr, int* total_cells_ptr, const char* parameters_file, const char* obstacles_file);
int serialise(const parameters parameters, const grid* old_grid, const char* obstacles, const float* average_velocity);
int finalise(parameters* parameters_ptr, grid** old_grid_ptr, grid** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr);

// main - initialise, realise, serialise, finalise
int main(int argc, char* argv[])
{
  parameters parameters;          // simulation parameters
  grid*  old_grid         = NULL; // old grid of cells
  grid*  new_grid         = NULL; // new grid of cells
  grid*  tmp_grid         = NULL; // tmp grid of cells
  char*  obstacles        = NULL; // grid of obstacles
  float* average_velocity = NULL; // average velocity
  int    total_cells;             // total cells
  float  reynolds_number;         // reynolds number
  char*  parameters_file  = NULL; // name of parameters file
  char*  obstacles_file   = NULL; // name of obstacles file
  struct timeval time;            // structure to hold elapsed time
  double tic, toc;                // float to record elapsed time

  // parse command line arguments
  parameters_file = argv[1];
  obstacles_file = argv[2];

  // load data structures
  initialise(&parameters, &old_grid, &new_grid, &obstacles, &average_velocity, &total_cells, parameters_file, obstacles_file);

  // get initial time
  gettimeofday(&time, NULL);
  tic = time.tv_sec + (time.tv_usec / 1000000.0);

  // iterate timestep
  for (int tt = 0; tt < parameters.iterations; tt++)
  {
    average_velocity[tt] = realise(parameters, old_grid, new_grid, obstacles, total_cells);
    tmp_grid = old_grid;
    old_grid = new_grid;
    new_grid = tmp_grid;
  }

  // get final time
  gettimeofday(&time, NULL);
  toc = time.tv_sec + (time.tv_usec / 1000000.0);
  
  // compute reynolds number
  reynolds_number = average_velocity[parameters.iterations - 1] * parameters.reynolds / (1.f / 6.f * (2.f / parameters.relaxation - 1.f));  

  // write performance metrics
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Reynolds number:\t\t%.12E\n", reynolds_number);

  // write final values
  serialise(parameters, old_grid, obstacles, average_velocity);

  // free data structures
  finalise(&parameters, &old_grid, &new_grid, &obstacles, &average_velocity);

  return EXIT_SUCCESS;
}

// realise - accelerate, propogate, collide, rebound
float realise(const parameters parameters, grid* restrict old_grid, grid* restrict new_grid, const char* restrict obstacles, const int total_cells)
{
  float total_velocity = 0.f; // total velocity

  const float a1 = parameters.density * parameters.acceleration / 9.f;  // acceleration factor
  const float a2 = parameters.density * parameters.acceleration / 36.f; // acceleration factor

  const float w0 = 4.f / 9.f;  // weighting factor
  const float w1 = 1.f / 9.f;  // weighting factor
  const float w2 = 1.f / 36.f; // weighting factor

  int jj = parameters.ny - 2;
  
  // loop over row
  #pragma omp simd
  for (int ii = 0; ii < parameters.nx; ii++)
  {
    // accelerate flow of unoccupied cells
    if (!obstacles[ii + jj * parameters.nx] && (old_grid->s3[ii + jj * parameters.nx] - a1) > 0.f && (old_grid->s6[ii + jj * parameters.nx] - a2) > 0.f && (old_grid->s7[ii + jj * parameters.nx] - a2) > 0.f)
    {
      // increase east-side speeds
      old_grid->s1[ii + jj * parameters.nx] += a1;
      old_grid->s5[ii + jj * parameters.nx] += a2;
      old_grid->s8[ii + jj * parameters.nx] += a2;
      // decrease west-side speeds
      old_grid->s3[ii + jj * parameters.nx] -= a1;
      old_grid->s6[ii + jj * parameters.nx] -= a2;
      old_grid->s7[ii + jj * parameters.nx] -= a2;
    }
  }
  
  // loop over grid
  #pragma omp parallel for reduction(+:total_velocity) num_threads(THREADS) proc_bind(BINDING) schedule(SCHEDULE)
  for (int jj = 0; jj < parameters.ny; jj++)
  {
    #pragma omp simd reduction(+:total_velocity)
    for (int ii = 0; ii < parameters.nx; ii++)
    {
      const int xe = (ii == parameters.nx - 1) ? (0) : (ii + 1);
      const int yn = (jj == parameters.ny - 1) ? (0) : (jj + 1);
      const int xw = (ii == 0) ? (ii + parameters.nx - 1) : (ii - 1);
      const int ys = (jj == 0) ? (jj + parameters.ny - 1) : (jj - 1);

      // propogate flow from neighbouring cells
      const float s0 = old_grid->s0[ii + jj * parameters.nx]; // centre
      const float s1 = old_grid->s1[xw + jj * parameters.nx]; // east
      const float s2 = old_grid->s2[ii + ys * parameters.nx]; // north
      const float s3 = old_grid->s3[xe + jj * parameters.nx]; // west
      const float s4 = old_grid->s4[ii + yn * parameters.nx]; // south
      const float s5 = old_grid->s5[xw + ys * parameters.nx]; // north-east
      const float s6 = old_grid->s6[xe + ys * parameters.nx]; // north-west
      const float s7 = old_grid->s7[xe + yn * parameters.nx]; // south-west
      const float s8 = old_grid->s8[xw + yn * parameters.nx]; // south-east

      // rebound flow of occupied cells
      if (obstacles[ii + jj * parameters.nx])
      {
        new_grid->s1[ii + jj * parameters.nx] = s3;
        new_grid->s2[ii + jj * parameters.nx] = s4;
        new_grid->s3[ii + jj * parameters.nx] = s1;
        new_grid->s4[ii + jj * parameters.nx] = s2;
        new_grid->s5[ii + jj * parameters.nx] = s7;
        new_grid->s6[ii + jj * parameters.nx] = s8;
        new_grid->s7[ii + jj * parameters.nx] = s5;
        new_grid->s8[ii + jj * parameters.nx] = s6;
      }
      // collide flow of unoccupied cells
      else
      {
        // compute local density
        const float l0 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8;

        // compute axial velocities
        const float vx = (s1 + s5 + s8 - (s3 + s6 + s7)) / l0; // x velocity component
        const float vy = (s2 + s5 + s6 - (s4 + s7 + s8)) / l0; // y velocity component
        const float v0 = vx * vx + vy * vy;                    // velocity squared

        // compute directional velocities
        const float v1 =   vx;      // east
        const float v2 =        vy; // north
        const float v3 = - vx;      // west
        const float v4 =      - vy; // south
        const float v5 =   vx + vy; // north-east
        const float v6 = - vx + vy; // north-west
        const float v7 = - vx - vy; // south-west
        const float v8 =   vx - vy; // south-east

        // compute equilibrium densities        
        const float d0 = w0 * l0 * (1.f - v0 * 1.5f);
        const float d1 = w1 * l0 * (1.f + v1 * 3.f + (v1 * v1) * 4.5f - v0 * 1.5f);
        const float d2 = w1 * l0 * (1.f + v2 * 3.f + (v2 * v2) * 4.5f - v0 * 1.5f);
        const float d3 = w1 * l0 * (1.f + v3 * 3.f + (v3 * v3) * 4.5f - v0 * 1.5f);
        const float d4 = w1 * l0 * (1.f + v4 * 3.f + (v4 * v4) * 4.5f - v0 * 1.5f);
        const float d5 = w2 * l0 * (1.f + v5 * 3.f + (v5 * v5) * 4.5f - v0 * 1.5f);
        const float d6 = w2 * l0 * (1.f + v6 * 3.f + (v6 * v6) * 4.5f - v0 * 1.5f);
        const float d7 = w2 * l0 * (1.f + v7 * 3.f + (v7 * v7) * 4.5f - v0 * 1.5f);
        const float d8 = w2 * l0 * (1.f + v8 * 3.f + (v8 * v8) * 4.5f - v0 * 1.5f);
        
        // compute relaxation
        new_grid->s0[ii + jj * parameters.nx] = s0 + parameters.relaxation * (d0 - s0);
        new_grid->s1[ii + jj * parameters.nx] = s1 + parameters.relaxation * (d1 - s1);
        new_grid->s2[ii + jj * parameters.nx] = s2 + parameters.relaxation * (d2 - s2);
        new_grid->s3[ii + jj * parameters.nx] = s3 + parameters.relaxation * (d3 - s3);
        new_grid->s4[ii + jj * parameters.nx] = s4 + parameters.relaxation * (d4 - s4);
        new_grid->s5[ii + jj * parameters.nx] = s5 + parameters.relaxation * (d5 - s5);
        new_grid->s6[ii + jj * parameters.nx] = s6 + parameters.relaxation * (d6 - s6);
        new_grid->s7[ii + jj * parameters.nx] = s7 + parameters.relaxation * (d7 - s7);
        new_grid->s8[ii + jj * parameters.nx] = s8 + parameters.relaxation * (d8 - s8);

        total_velocity += sqrtf(v0);
      }
    }
  }

  return total_velocity / total_cells;
}

// initialise - open, allocate, initialise, close
int initialise(parameters* parameters_ptr, grid** old_grid_ptr, grid** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr, int* total_cells_ptr, const char* parameters_file, const char* obstacles_file)
{
  char  message[1024]; // message buffer
  int   occupied;      // cell occupation
  int   retval;        // return values
  int   ii, jj;        // array indices
  FILE* fp;            // file pointer

  // open the parameters file
  fp = fopen(parameters_file, "r");

  // read the parameters values
  retval = fscanf(fp, "%d\n", &(parameters_ptr->nx));
  retval = fscanf(fp, "%d\n", &(parameters_ptr->ny));
  retval = fscanf(fp, "%d\n", &(parameters_ptr->reynolds));
  retval = fscanf(fp, "%d\n", &(parameters_ptr->iterations));
  retval = fscanf(fp, "%f\n", &(parameters_ptr->density));
  retval = fscanf(fp, "%f\n", &(parameters_ptr->relaxation));
  retval = fscanf(fp, "%f\n", &(parameters_ptr->acceleration));

  // close the parameters file
  fclose(fp);

  // allocate old grid
  *old_grid_ptr = _mm_malloc(sizeof(grid), 64);
  (*old_grid_ptr)->s0 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s1 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s2 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s3 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s4 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s5 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s6 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s7 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*old_grid_ptr)->s8 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  
  // allocate new grid
  *new_grid_ptr = _mm_malloc(sizeof(grid), 64);
  (*new_grid_ptr)->s0 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s1 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s2 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s3 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s4 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s5 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s6 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s7 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  (*new_grid_ptr)->s8 = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);
  
  // allocate obstacles
  *obstacles_ptr = _mm_malloc(sizeof(char) * parameters_ptr->nx * parameters_ptr->ny, 64);

  // initialise weighting factors
  const float w0 = parameters_ptr->density * 4.f / 9.f;
  const float w1 = parameters_ptr->density       / 9.f;
  const float w2 = parameters_ptr->density       / 36.f;

  // initialise grids
  #pragma omp parallel for num_threads(THREADS) proc_bind(BINDING) schedule(SCHEDULE)
  for (int jj = 0; jj < parameters_ptr->ny; jj++)
  {
    for (int ii = 0; ii < parameters_ptr->nx; ii++)
    {
      (*old_grid_ptr)->s0[ii + jj * parameters_ptr->nx] = w0;
      (*old_grid_ptr)->s1[ii + jj * parameters_ptr->nx] = w1;
      (*old_grid_ptr)->s2[ii + jj * parameters_ptr->nx] = w1;
      (*old_grid_ptr)->s3[ii + jj * parameters_ptr->nx] = w1;
      (*old_grid_ptr)->s4[ii + jj * parameters_ptr->nx] = w1;
      (*old_grid_ptr)->s5[ii + jj * parameters_ptr->nx] = w2;
      (*old_grid_ptr)->s6[ii + jj * parameters_ptr->nx] = w2;
      (*old_grid_ptr)->s7[ii + jj * parameters_ptr->nx] = w2;
      (*old_grid_ptr)->s8[ii + jj * parameters_ptr->nx] = w2;
      (*new_grid_ptr)->s0[ii + jj * parameters_ptr->nx] = w0;
      (*new_grid_ptr)->s1[ii + jj * parameters_ptr->nx] = w1;
      (*new_grid_ptr)->s2[ii + jj * parameters_ptr->nx] = w1;
      (*new_grid_ptr)->s3[ii + jj * parameters_ptr->nx] = w1;
      (*new_grid_ptr)->s4[ii + jj * parameters_ptr->nx] = w1;
      (*new_grid_ptr)->s5[ii + jj * parameters_ptr->nx] = w2;
      (*new_grid_ptr)->s6[ii + jj * parameters_ptr->nx] = w2;
      (*new_grid_ptr)->s7[ii + jj * parameters_ptr->nx] = w2;
      (*new_grid_ptr)->s8[ii + jj * parameters_ptr->nx] = w2;
      (*obstacles_ptr)[ii + jj * parameters_ptr->nx] = 0;
    }
  }

  // open the obstacles file
  fp = fopen(obstacles_file, "r");

  // read the obstacles values
  while ((retval = fscanf(fp, "%d %d %d\n", &ii, &jj, &occupied)) != EOF)
  {
    (*obstacles_ptr)[ii + jj * parameters_ptr->nx] = occupied;
  }

  // close the obstacles file
  fclose(fp);

  // allocate average velocity
  *average_velocity_ptr = _mm_malloc(sizeof(float) * parameters_ptr->iterations, 64);

  // initialise total cells
  *total_cells_ptr = 0.f;
  for (int jj = 0; jj < parameters_ptr->ny; jj++)
  {
    for (int ii = 0; ii < parameters_ptr->nx; ii++)
    {
      if (!(*obstacles_ptr)[ii + jj * parameters_ptr->nx])
      {
        *total_cells_ptr += 1.f;
      }
    }
  }

  return EXIT_SUCCESS;
}

// write - final state, average velocity
int serialise(const parameters parameters, const grid* old_grid, const char* obstacles, const float* average_velocity)
{
  const float c0 = 1.f / 3.f; // speed factor
  float l0;                   // local density
  float p0;                   // local pressure
  float vx;                   // x velocity component
  float vy;                   // y velocity component
  float v0;                   // velocity combination
  FILE* fp;                   // file pointer

  // open the final state file
  fp = fopen(FSFILE, "w");

  // write the final state values
  for (int jj = 0; jj < parameters.ny; jj++)
  {
    for (int ii = 0; ii < parameters.nx; ii++)
    {
      if (obstacles[ii + jj * parameters.nx])
      {
        vx = vy = v0 = 0.f;
        p0 = parameters.density * c0;
      }
      else
      {
        l0 = old_grid->s0[ii + jj * parameters.nx] + old_grid->s1[ii + jj * parameters.nx] + old_grid->s2[ii + jj * parameters.nx] + old_grid->s3[ii + jj * parameters.nx] + old_grid->s4[ii + jj * parameters.nx] + old_grid->s5[ii + jj * parameters.nx] + old_grid->s6[ii + jj * parameters.nx] + old_grid->s7[ii + jj * parameters.nx] + old_grid->s8[ii + jj * parameters.nx];
        vx = (old_grid->s1[ii + jj * parameters.nx] + old_grid->s5[ii + jj * parameters.nx] + old_grid->s8[ii + jj * parameters.nx] - (old_grid->s3[ii + jj * parameters.nx] + old_grid->s6[ii + jj * parameters.nx] + old_grid->s7[ii + jj * parameters.nx])) / l0;
        vy = (old_grid->s2[ii + jj * parameters.nx] + old_grid->s5[ii + jj * parameters.nx] + old_grid->s6[ii + jj * parameters.nx] - (old_grid->s4[ii + jj * parameters.nx] + old_grid->s7[ii + jj * parameters.nx] + old_grid->s8[ii + jj * parameters.nx])) / l0;
        v0 = sqrtf((vx * vx) + (vy * vy));
        p0 = l0 * c0;
      }

      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, vx, vy, v0, p0, obstacles[ii + jj * parameters.nx]);
    }
  }

  // close the final state file
  fclose(fp);

  // open the average velocity file
  fp = fopen(AVFILE, "w");

  // write the average velocity values
  for (int tt = 0; tt < parameters.iterations; tt++)
  {
    fprintf(fp, "%d:\t%.12E\n", tt, average_velocity[tt]);
  }

  // close the average velocity file
  fclose(fp);

  return EXIT_SUCCESS;
}

// finalise - free allocated memory
int finalise(parameters* parameters_ptr, grid** old_grid_ptr, grid** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr)
{
  _mm_free((*old_grid_ptr)->s0);
  (*old_grid_ptr)->s0 = NULL;
  _mm_free((*old_grid_ptr)->s1);
  (*old_grid_ptr)->s1 = NULL;
  _mm_free((*old_grid_ptr)->s2);
  (*old_grid_ptr)->s2 = NULL;
  _mm_free((*old_grid_ptr)->s3);
  (*old_grid_ptr)->s3 = NULL;
  _mm_free((*old_grid_ptr)->s4);
  (*old_grid_ptr)->s4 = NULL;
  _mm_free((*old_grid_ptr)->s5);
  (*old_grid_ptr)->s5 = NULL;
  _mm_free((*old_grid_ptr)->s6);
  (*old_grid_ptr)->s6 = NULL;
  _mm_free((*old_grid_ptr)->s7);
  (*old_grid_ptr)->s7 = NULL;
  _mm_free((*old_grid_ptr)->s8);
  (*old_grid_ptr)->s8 = NULL;
  _mm_free(*old_grid_ptr);
  *old_grid_ptr = NULL;

  _mm_free((*new_grid_ptr)->s0);
  (*new_grid_ptr)->s0 = NULL;
  _mm_free((*new_grid_ptr)->s1);
  (*new_grid_ptr)->s1 = NULL;
  _mm_free((*new_grid_ptr)->s2);
  (*new_grid_ptr)->s2 = NULL;
  _mm_free((*new_grid_ptr)->s3);
  (*new_grid_ptr)->s3 = NULL;
  _mm_free((*new_grid_ptr)->s4);
  (*new_grid_ptr)->s4 = NULL;
  _mm_free((*new_grid_ptr)->s5);
  (*new_grid_ptr)->s5 = NULL;
  _mm_free((*new_grid_ptr)->s6);
  (*new_grid_ptr)->s6 = NULL;
  _mm_free((*new_grid_ptr)->s7);
  (*new_grid_ptr)->s7 = NULL;
  _mm_free((*new_grid_ptr)->s8);
  (*new_grid_ptr)->s8 = NULL;
  _mm_free(*new_grid_ptr);
  *new_grid_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*average_velocity_ptr);
  *average_velocity_ptr = NULL;

  return EXIT_SUCCESS;
}