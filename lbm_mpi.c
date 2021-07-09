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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS 9
#define MASTER  0
#define FSFILE "outputs/final_state.data"
#define AVFILE "outputs/average_velocity.data"

// structure to hold parameters
typedef struct
{
  int   size, rank;   // MPI size and rank
  int   above, below; // MPI adjacent rank
  int   gx, gy;       // global cells in each direction
  int   lx, ly;       // local cells in each direction
  int   rx, ry;       // remote cells in each direction
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
float realise(const parameters parameters, grid* restrict local_old_grid, grid* restrict local_new_grid, float* restrict halo_snd_grid, float* restrict halo_rcv_grid, const char* restrict local_obstacles);
int initialise(parameters* parameters_ptr, grid** global_grid_ptr, grid** local_old_grid_ptr, grid** local_new_grid_ptr, float** halo_snd_grid_ptr, float** halo_rcv_grid_ptr, char** global_obstacles_ptr, char** local_obstacles_ptr, float** global_average_velocity_ptr, float** local_total_velocity_ptr, int* total_cells_ptr, const char* parameters_file, const char* obstacles_file);
int localise(const parameters parameters, const int adjacent, const int previous);
int globalise(const parameters parameters, grid* global_grid, grid* local_old_grid, char* global_obstacles, float* global_average_velocity, float* local_total_velocity, int total_cells);
int serialise(const parameters parameters, const grid* global_grid, const char* global_obstacles, const float* global_average_velocity);
int finalise(parameters* parameters_ptr, grid** global_grid_ptr, grid** local_old_grid_ptr, grid** local_new_grid_ptr, float** halo_snd_grid_ptr, float** halo_rcv_grid_ptr, char** global_obstacles_ptr, char** local_obstacles_ptr, float** global_average_velocity_ptr, float** local_total_velocity_ptr);

// main - initialise, localise, globalise, realise, serialise, finalise
int main(int argc, char* argv[])
{
  parameters parameters;                 // simulation parameters
  grid*  global_grid             = NULL; // global grid of cells
  grid*  local_old_grid          = NULL; // local old grid of cells
  grid*  local_new_grid          = NULL; // local new grid of cells
  grid*  local_tmp_grid          = NULL; // local tmp grid of cells
  float* halo_snd_grid           = NULL; // halo snd grid of cells
  float* halo_rcv_grid           = NULL; // halo rcv grid of cells
  char*  global_obstacles        = NULL; // global grid of obstacles
  char*  local_obstacles         = NULL; // local grid of obstacles
  float* global_average_velocity = NULL; // global average velocity
  float* local_total_velocity    = NULL; // local total velocity
  int    total_cells;                    // total cells
  float  reynolds_number;                // reynolds number
  char*  parameters_file         = NULL; // name of parameters file
  char*  obstacles_file          = NULL; // name of obstacles file
  struct timeval time;                   // structure to hold elapsed time
  double tic, toc;                       // float to record elapsed time

  // initialise MPI environment
  MPI_Init(&argc, &argv);

  // parse command line arguments
  parameters_file = argv[1];
  obstacles_file = argv[2];

  // get initial time
  gettimeofday(&time, NULL);
  tic = time.tv_sec + (time.tv_usec / 1000000.0);

  // load data structures
  initialise(&parameters, &global_grid, &local_old_grid, &local_new_grid, &halo_snd_grid, &halo_rcv_grid, &global_obstacles, &local_obstacles, &global_average_velocity, &local_total_velocity, &total_cells, parameters_file, obstacles_file);

  // iterate timestep
  for (int tt = 0; tt < parameters.iterations; tt++)
  {
    local_total_velocity[tt] = realise(parameters, local_old_grid, local_new_grid, halo_snd_grid, halo_rcv_grid, local_obstacles);
    local_tmp_grid = local_old_grid;
    local_old_grid = local_new_grid;
    local_new_grid = local_tmp_grid;
  }

  // collate data structures
  globalise(parameters, global_grid, local_old_grid, global_obstacles, global_average_velocity, local_total_velocity, total_cells);

  // get final time
  gettimeofday(&time, NULL);
  toc = time.tv_sec + (time.tv_usec / 1000000.0);

  // compute reynolds number
  if (parameters.rank == MASTER) reynolds_number = global_average_velocity[parameters.iterations - 1] * parameters.reynolds / (1.f / 6.f * (2.f / parameters.relaxation - 1.f));  

  // write performance metrics
  if (parameters.rank == MASTER)
  {
    printf("Elapsed time:\t\t\t%.6lf (s)\n",   toc  - tic);
    printf("Reynolds number:\t\t%.12E\n", reynolds_number);
  }
  
  // write final values
  serialise(parameters, global_grid, global_obstacles, global_average_velocity);

  // free data structures
  finalise(&parameters, &global_grid, &local_old_grid, &local_new_grid, &halo_snd_grid, &halo_rcv_grid, &global_obstacles, &local_obstacles, &global_average_velocity, &local_total_velocity);

  // finalise MPI environment
  MPI_Finalize();

  return EXIT_SUCCESS;
}

// realise - accelerate, propogate, collide, rebound
float realise(const parameters parameters, grid* restrict local_old_grid, grid* restrict local_new_grid, float* restrict halo_snd_grid, float* restrict halo_rcv_grid, const char* restrict local_obstacles)
{
  float total_velocity = 0.f; // total velocity

  const float a1 = parameters.density * parameters.acceleration / 9.f;  // acceleration factor
  const float a2 = parameters.density * parameters.acceleration / 36.f; // acceleration factor

  const float w0 = 4.f / 9.f;  // weighting factor
  const float w1 = 1.f / 9.f;  // weighting factor
  const float w2 = 1.f / 36.f; // weighting factor

  int jj = parameters.gy - parameters.ry - 1;
  if (parameters.ry - 1 == parameters.gy - 2) jj = 0;
  if (parameters.ry + parameters.ly == parameters.gy - 2) jj = parameters.ly + 1;

  // loop over row
  if ((parameters.ry <= parameters.gy - 2 && parameters.ry + parameters.ly > parameters.gy - 2) || (parameters.ry - 1 == parameters.gy - 2) || (parameters.ry + parameters.ly == parameters.gy - 2))
  {
    #pragma omp simd
    for (int ii = 0; ii < parameters.lx; ii++)
    {
      // accelerate flow of unoccupied cells
      if (!local_obstacles[ii + jj * parameters.lx] && (local_old_grid->s3[ii + jj * parameters.lx] - a1) > 0.f && (local_old_grid->s6[ii + jj * parameters.lx] - a2) > 0.f && (local_old_grid->s7[ii + jj * parameters.lx] - a2) > 0.f)
      {
        // increase east-side speeds
        local_old_grid->s1[ii + jj * parameters.lx] += a1;
        local_old_grid->s5[ii + jj * parameters.lx] += a2;
        local_old_grid->s8[ii + jj * parameters.lx] += a2;
        // decrease west-side speeds
        local_old_grid->s3[ii + jj * parameters.lx] -= a1;
        local_old_grid->s6[ii + jj * parameters.lx] -= a2;
        local_old_grid->s7[ii + jj * parameters.lx] -= a2;
      }
    }
  }

  // loop over grid
  for (int jj = 1; jj < parameters.ly + 1; jj++)
  {
    #pragma omp simd reduction(+:total_velocity)
    for (int ii = 0; ii < parameters.lx; ii++)
    {
      const int xe = (ii == parameters.lx - 1) ? (0) : (ii + 1);
      const int yn = (jj + 1);
      const int xw = (ii == 0) ? (ii + parameters.lx - 1) : (ii - 1);
      const int ys = (jj - 1);

      // propogate flow from neighbouring cells
      const float s0 = local_old_grid->s0[ii + jj * parameters.lx]; // centre
      const float s1 = local_old_grid->s1[xw + jj * parameters.lx]; // east
      const float s2 = local_old_grid->s2[ii + ys * parameters.lx]; // north
      const float s3 = local_old_grid->s3[xe + jj * parameters.lx]; // west
      const float s4 = local_old_grid->s4[ii + yn * parameters.lx]; // south
      const float s5 = local_old_grid->s5[xw + ys * parameters.lx]; // north-east
      const float s6 = local_old_grid->s6[xe + ys * parameters.lx]; // north-west
      const float s7 = local_old_grid->s7[xe + yn * parameters.lx]; // south-west
      const float s8 = local_old_grid->s8[xw + yn * parameters.lx]; // south-east

      // rebound flow of occupied cells
      if (local_obstacles[ii + jj * parameters.lx])
      {
        local_new_grid->s1[ii + jj * parameters.lx] = s3;
        local_new_grid->s2[ii + jj * parameters.lx] = s4;
        local_new_grid->s3[ii + jj * parameters.lx] = s1;
        local_new_grid->s4[ii + jj * parameters.lx] = s2;
        local_new_grid->s5[ii + jj * parameters.lx] = s7;
        local_new_grid->s6[ii + jj * parameters.lx] = s8;
        local_new_grid->s7[ii + jj * parameters.lx] = s5;
        local_new_grid->s8[ii + jj * parameters.lx] = s6;
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
        local_new_grid->s0[ii + jj * parameters.lx] = s0 + parameters.relaxation * (d0 - s0);
        local_new_grid->s1[ii + jj * parameters.lx] = s1 + parameters.relaxation * (d1 - s1);
        local_new_grid->s2[ii + jj * parameters.lx] = s2 + parameters.relaxation * (d2 - s2);
        local_new_grid->s3[ii + jj * parameters.lx] = s3 + parameters.relaxation * (d3 - s3);
        local_new_grid->s4[ii + jj * parameters.lx] = s4 + parameters.relaxation * (d4 - s4);
        local_new_grid->s5[ii + jj * parameters.lx] = s5 + parameters.relaxation * (d5 - s5);
        local_new_grid->s6[ii + jj * parameters.lx] = s6 + parameters.relaxation * (d6 - s6);
        local_new_grid->s7[ii + jj * parameters.lx] = s7 + parameters.relaxation * (d7 - s7);
        local_new_grid->s8[ii + jj * parameters.lx] = s8 + parameters.relaxation * (d8 - s8);

        total_velocity += sqrtf(v0);
      }
    }
  }

  // send to below, receive from above
  for (int ii = 0; ii < parameters.lx; ii++)
  {
    halo_snd_grid[ii + 0 * parameters.lx] = local_new_grid->s0[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 1 * parameters.lx] = local_new_grid->s1[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 2 * parameters.lx] = local_new_grid->s2[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 3 * parameters.lx] = local_new_grid->s3[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 4 * parameters.lx] = local_new_grid->s4[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 5 * parameters.lx] = local_new_grid->s5[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 6 * parameters.lx] = local_new_grid->s6[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 7 * parameters.lx] = local_new_grid->s7[ii + 1 * parameters.lx];
    halo_snd_grid[ii + 8 * parameters.lx] = local_new_grid->s8[ii + 1 * parameters.lx];
  }

  MPI_Sendrecv(halo_snd_grid, parameters.lx * NSPEEDS, MPI_FLOAT, parameters.below, 0, halo_rcv_grid, parameters.lx * NSPEEDS, MPI_FLOAT, parameters.above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int ii = 0; ii < parameters.lx; ii++)
  {
    local_new_grid->s0[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 0 * parameters.lx];
    local_new_grid->s1[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 1 * parameters.lx];
    local_new_grid->s2[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 2 * parameters.lx];
    local_new_grid->s3[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 3 * parameters.lx];
    local_new_grid->s4[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 4 * parameters.lx];
    local_new_grid->s5[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 5 * parameters.lx];
    local_new_grid->s6[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 6 * parameters.lx];
    local_new_grid->s7[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 7 * parameters.lx];
    local_new_grid->s8[ii + (parameters.ly + 1) * parameters.lx] = halo_rcv_grid[ii + 8 * parameters.lx];
  }
  
  // send to above, receive from below
  for (int ii = 0; ii < parameters.lx; ii++)
  {
    halo_snd_grid[ii + 0 * parameters.lx] = local_new_grid->s0[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 1 * parameters.lx] = local_new_grid->s1[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 2 * parameters.lx] = local_new_grid->s2[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 3 * parameters.lx] = local_new_grid->s3[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 4 * parameters.lx] = local_new_grid->s4[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 5 * parameters.lx] = local_new_grid->s5[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 6 * parameters.lx] = local_new_grid->s6[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 7 * parameters.lx] = local_new_grid->s7[ii + parameters.ly * parameters.lx];
    halo_snd_grid[ii + 8 * parameters.lx] = local_new_grid->s8[ii + parameters.ly * parameters.lx];
  }

  MPI_Sendrecv(halo_snd_grid, parameters.lx * NSPEEDS, MPI_FLOAT, parameters.above, 0, halo_rcv_grid, parameters.lx * NSPEEDS, MPI_FLOAT, parameters.below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int ii = 0; ii < parameters.lx; ii++)
  {
    local_new_grid->s0[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 0 * parameters.lx];
    local_new_grid->s1[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 1 * parameters.lx];
    local_new_grid->s2[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 2 * parameters.lx];
    local_new_grid->s3[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 3 * parameters.lx];
    local_new_grid->s4[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 4 * parameters.lx];
    local_new_grid->s5[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 5 * parameters.lx];
    local_new_grid->s6[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 6 * parameters.lx];
    local_new_grid->s7[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 7 * parameters.lx];
    local_new_grid->s8[ii + 0 * parameters.lx] = halo_rcv_grid[ii + 8 * parameters.lx];
  }

  return total_velocity;
}

// initialise - open, allocate, initialise, close
int initialise(parameters* parameters_ptr, grid** global_grid_ptr, grid** local_old_grid_ptr, grid** local_new_grid_ptr, float** halo_snd_grid_ptr, float** halo_rcv_grid_ptr, char** global_obstacles_ptr, char** local_obstacles_ptr, float** global_average_velocity_ptr, float** local_total_velocity_ptr, int* total_cells_ptr, const char* parameters_file, const char* obstacles_file)
{
  char  message[1024]; // message buffer
  int   occupied;      // cell occupation
  int   retval;        // return values
  int   ii, jj;        // array indices
  FILE* fp;            // file pointer

  // open the parameters file
  fp = fopen(parameters_file, "r");

  // read the parameters values
  MPI_Comm_size(MPI_COMM_WORLD, &(parameters_ptr->size));
  MPI_Comm_rank(MPI_COMM_WORLD, &(parameters_ptr->rank));
  parameters_ptr->above = (parameters_ptr->rank == parameters_ptr->size - 1) ? (0) : (parameters_ptr->rank + 1);
  parameters_ptr->below = (parameters_ptr->rank == 0) ? (parameters_ptr->rank + parameters_ptr->size - 1) : (parameters_ptr->rank - 1); 
  retval = fscanf(fp, "%d\n", &(parameters_ptr->gx));
  retval = fscanf(fp, "%d\n", &(parameters_ptr->gy));
  parameters_ptr->lx = parameters_ptr->gx;
  parameters_ptr->ly = localise(*parameters_ptr, 0, 0);
  parameters_ptr->rx = parameters_ptr->gx;
  parameters_ptr->ry = localise(*parameters_ptr, 0, 1);
  retval = fscanf(fp, "%d\n", &(parameters_ptr->reynolds));
  retval = fscanf(fp, "%d\n", &(parameters_ptr->iterations));
  retval = fscanf(fp, "%f\n", &(parameters_ptr->density));
  retval = fscanf(fp, "%f\n", &(parameters_ptr->relaxation));
  retval = fscanf(fp, "%f\n", &(parameters_ptr->acceleration));

  // close the parameters file
  fclose(fp);

  // allocate global grid
  if (parameters_ptr->rank == MASTER)
  {
    *global_grid_ptr = _mm_malloc(sizeof(grid), 64);
    (*global_grid_ptr)->s0 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s1 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s2 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s3 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s4 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s5 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s6 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s7 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
    (*global_grid_ptr)->s8 = _mm_malloc(sizeof(float) * parameters_ptr->gx * parameters_ptr->gy, 64);
  }

  // allocate local old grid
  *local_old_grid_ptr = _mm_malloc(sizeof(grid), 64);
  (*local_old_grid_ptr)->s0 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s1 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s2 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s3 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s4 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s5 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s6 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s7 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_old_grid_ptr)->s8 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  
  // allocate local new grid
  *local_new_grid_ptr = _mm_malloc(sizeof(grid), 64);
  (*local_new_grid_ptr)->s0 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s1 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s2 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s3 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s4 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s5 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s6 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s7 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);
  (*local_new_grid_ptr)->s8 = _mm_malloc(sizeof(float) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);

  // allocate halo snd grid
  *halo_snd_grid_ptr = _mm_malloc(sizeof(float) * parameters_ptr->lx * NSPEEDS, 64);

  // allocate halo rcv grid
  *halo_rcv_grid_ptr = _mm_malloc(sizeof(float) * parameters_ptr->lx * NSPEEDS, 64);

  // allocate global obstacles
  if (parameters_ptr->rank == MASTER)
  {
    *global_obstacles_ptr = _mm_malloc(sizeof(char) * parameters_ptr->gx * parameters_ptr->gy, 64);
  }

  // allocate local obstacles
  *local_obstacles_ptr = _mm_malloc(sizeof(char) * parameters_ptr->lx * (parameters_ptr->ly + 2), 64);

  // initialise weighting factors
  const float w0 = parameters_ptr->density * 4.f / 9.f;
  const float w1 = parameters_ptr->density       / 9.f;
  const float w2 = parameters_ptr->density       / 36.f;

  // initialise global grids
  if (parameters_ptr->rank == MASTER)
  {
    for (int jj = 0; jj < parameters_ptr->gy; jj++)
    {
      for (int ii = 0; ii < parameters_ptr->gx; ii++)
      {
        (*global_grid_ptr)->s0[ii + jj * parameters_ptr->gx] = w0;
        (*global_grid_ptr)->s1[ii + jj * parameters_ptr->gx] = w1;
        (*global_grid_ptr)->s2[ii + jj * parameters_ptr->gx] = w1;
        (*global_grid_ptr)->s3[ii + jj * parameters_ptr->gx] = w1;
        (*global_grid_ptr)->s4[ii + jj * parameters_ptr->gx] = w1;
        (*global_grid_ptr)->s5[ii + jj * parameters_ptr->gx] = w2;
        (*global_grid_ptr)->s6[ii + jj * parameters_ptr->gx] = w2;
        (*global_grid_ptr)->s7[ii + jj * parameters_ptr->gx] = w2;
        (*global_grid_ptr)->s8[ii + jj * parameters_ptr->gx] = w2;
        (*global_obstacles_ptr)[ii + jj * parameters_ptr->gx] = 0;
      }
    }
  }

  // initialise local grids
  for (int jj = 0; jj < (parameters_ptr->ly + 2); jj++)
  {
    for (int ii = 0; ii < parameters_ptr->lx; ii++)
    {
      (*local_old_grid_ptr)->s0[ii + jj * parameters_ptr->lx] = w0;
      (*local_old_grid_ptr)->s1[ii + jj * parameters_ptr->lx] = w1;
      (*local_old_grid_ptr)->s2[ii + jj * parameters_ptr->lx] = w1;
      (*local_old_grid_ptr)->s3[ii + jj * parameters_ptr->lx] = w1;
      (*local_old_grid_ptr)->s4[ii + jj * parameters_ptr->lx] = w1;
      (*local_old_grid_ptr)->s5[ii + jj * parameters_ptr->lx] = w2;
      (*local_old_grid_ptr)->s6[ii + jj * parameters_ptr->lx] = w2;
      (*local_old_grid_ptr)->s7[ii + jj * parameters_ptr->lx] = w2;
      (*local_old_grid_ptr)->s8[ii + jj * parameters_ptr->lx] = w2;
      (*local_new_grid_ptr)->s0[ii + jj * parameters_ptr->lx] = w0;
      (*local_new_grid_ptr)->s1[ii + jj * parameters_ptr->lx] = w1;
      (*local_new_grid_ptr)->s2[ii + jj * parameters_ptr->lx] = w1;
      (*local_new_grid_ptr)->s3[ii + jj * parameters_ptr->lx] = w1;
      (*local_new_grid_ptr)->s4[ii + jj * parameters_ptr->lx] = w1;
      (*local_new_grid_ptr)->s5[ii + jj * parameters_ptr->lx] = w2;
      (*local_new_grid_ptr)->s6[ii + jj * parameters_ptr->lx] = w2;
      (*local_new_grid_ptr)->s7[ii + jj * parameters_ptr->lx] = w2;
      (*local_new_grid_ptr)->s8[ii + jj * parameters_ptr->lx] = w2;
      (*local_obstacles_ptr)[ii + jj * parameters_ptr->lx] = 0;
    }
  }

  // open the obstacles file
  fp = fopen(obstacles_file, "r");

  // read the obstacles values
  while ((retval = fscanf(fp, "%d %d %d\n", &ii, &jj, &occupied)) != EOF)
  {
    // perform checks
    if (parameters_ptr->rank == MASTER) (*global_obstacles_ptr)[ii + jj * parameters_ptr->gx] = occupied;
    if (jj >= parameters_ptr->ry - 1 && jj < parameters_ptr->ry + parameters_ptr->ly + 1) (*local_obstacles_ptr)[ii + (jj - parameters_ptr->ry + 1) * parameters_ptr->gx] = occupied;
    if (jj == 0 && parameters_ptr->ry + parameters_ptr->ly == parameters_ptr->gy) (*local_obstacles_ptr)[ii + (parameters_ptr->ly + 1) * parameters_ptr->gx] = occupied;
    if (jj == parameters_ptr->gy - 1 && parameters_ptr->ry == 0) (*local_obstacles_ptr)[ii + 0 * parameters_ptr->gx] = occupied;
  }

  // close the obstacles file
  fclose(fp);

  // allocate global average velocity
  if (parameters_ptr->rank == MASTER) *global_average_velocity_ptr = _mm_malloc(sizeof(float) * parameters_ptr->iterations, 64);

  // allocate local total velocity
  *local_total_velocity_ptr = _mm_malloc(sizeof(float) * parameters_ptr->iterations, 64);

  // initialise total cells
  if (parameters_ptr->rank == MASTER) 
  {
    *total_cells_ptr = 0.f;
    for (int jj = 0; jj < parameters_ptr->gy; jj++)
    {
      for (int ii = 0; ii < parameters_ptr->gx; ii++)
      {
        if (!(*global_obstacles_ptr)[ii + jj * parameters_ptr->gx])
        {
          *total_cells_ptr += 1.f;
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

// localise - compute localised grid
int localise(const parameters parameters, const int adjacent, const int previous)
{
  int ly = 0;

  // evenly distribute rows among ranks
  for (int kk = (1 - previous) * (parameters.rank + adjacent); kk < (1 - previous) * 1 + (parameters.rank + adjacent); kk++)
  {
    ly += parameters.gy / parameters.size;
    if (kk < parameters.gy % parameters.size)
    {
      ly += 1;
    } 
  }

  return ly;
}

// globalise - collate globalised grid
int globalise(const parameters parameters, grid* global_grid, grid* local_old_grid, char* global_obstacles, float* global_average_velocity, float* local_total_velocity, int total_cells)
{
  // send to master rank
  if (parameters.rank == MASTER)
  {
    for (int jj = 1; jj < parameters.ly + 1; jj++)
    {
      for (int ii = 0; ii < parameters.lx; ii++)
      {
        global_grid->s0[ii + (jj - 1) * parameters.lx] = local_old_grid->s0[ii + jj * parameters.lx];
        global_grid->s1[ii + (jj - 1) * parameters.lx] = local_old_grid->s1[ii + jj * parameters.lx];
        global_grid->s2[ii + (jj - 1) * parameters.lx] = local_old_grid->s2[ii + jj * parameters.lx];
        global_grid->s3[ii + (jj - 1) * parameters.lx] = local_old_grid->s3[ii + jj * parameters.lx];
        global_grid->s4[ii + (jj - 1) * parameters.lx] = local_old_grid->s4[ii + jj * parameters.lx];
        global_grid->s5[ii + (jj - 1) * parameters.lx] = local_old_grid->s5[ii + jj * parameters.lx];
        global_grid->s6[ii + (jj - 1) * parameters.lx] = local_old_grid->s6[ii + jj * parameters.lx];
        global_grid->s7[ii + (jj - 1) * parameters.lx] = local_old_grid->s7[ii + jj * parameters.lx];
        global_grid->s8[ii + (jj - 1) * parameters.lx] = local_old_grid->s8[ii + jj * parameters.lx];
      }
    }
  }
  else
  {
    float* local_snd_grid = _mm_malloc(sizeof(float) * parameters.lx * parameters.ly * NSPEEDS, 64);
    for (int jj = 1; jj < parameters.ly + 1; jj++)
    {
      for (int ii = 0; ii < parameters.lx; ii++)
      {
        local_snd_grid[ii + (jj - 1) * parameters.lx + 0 * parameters.lx * parameters.ly] = local_old_grid->s0[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 1 * parameters.lx * parameters.ly] = local_old_grid->s1[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 2 * parameters.lx * parameters.ly] = local_old_grid->s2[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 3 * parameters.lx * parameters.ly] = local_old_grid->s3[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 4 * parameters.lx * parameters.ly] = local_old_grid->s4[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 5 * parameters.lx * parameters.ly] = local_old_grid->s5[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 6 * parameters.lx * parameters.ly] = local_old_grid->s6[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 7 * parameters.lx * parameters.ly] = local_old_grid->s7[ii + jj * parameters.lx];
        local_snd_grid[ii + (jj - 1) * parameters.lx + 8 * parameters.lx * parameters.ly] = local_old_grid->s8[ii + jj * parameters.lx];
      }
    }
    MPI_Ssend(local_snd_grid, parameters.lx * parameters.ly * NSPEEDS, MPI_FLOAT, MASTER, parameters.rank, MPI_COMM_WORLD);
    _mm_free(local_snd_grid);
    local_snd_grid = NULL;
  }

  // recieve from worker ranks
  if (parameters.rank == MASTER)
  {
    for (int kk = 1; kk < parameters.size; kk++)
    {
      int ly = localise(parameters, kk, 0);
      int ry = localise(parameters, kk, 1);
      float* local_rcv_grid = _mm_malloc(sizeof(float) * parameters.lx * ly * NSPEEDS, 64);
      MPI_Recv(local_rcv_grid, parameters.lx * ly * NSPEEDS, MPI_FLOAT, kk, kk, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int jj = 0; jj < ly; jj++)
      {
        for (int ii = 0; ii < parameters.lx; ii++)
        {
          global_grid->s0[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 0 * parameters.lx * ly];
          global_grid->s1[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 1 * parameters.lx * ly];
          global_grid->s2[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 2 * parameters.lx * ly];
          global_grid->s3[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 3 * parameters.lx * ly];
          global_grid->s4[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 4 * parameters.lx * ly];
          global_grid->s5[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 5 * parameters.lx * ly];
          global_grid->s6[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 6 * parameters.lx * ly];
          global_grid->s7[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 7 * parameters.lx * ly];
          global_grid->s8[ii + (ry + jj) * parameters.lx] = local_rcv_grid[ii + jj * parameters.lx + 8 * parameters.lx * ly];
        }
      }
      _mm_free(local_rcv_grid);
      local_rcv_grid = NULL;
    }
  }

  // sum local total velocities
  MPI_Reduce(local_total_velocity, global_average_velocity, parameters.iterations, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);

  // compute global average velocity
  if (parameters.rank == MASTER) 
  {
    for (int tt = 0; tt < parameters.iterations; tt++) 
    {
      global_average_velocity[tt] /= total_cells;
    }
  }

  return EXIT_SUCCESS;
}

// serialise - final state, average velocity
int serialise(const parameters parameters, const grid* global_grid, const char* global_obstacles, const float* global_average_velocity)
{
  const float c0 = 1.f / 3.f; // speed factor
  float l0;                   // local density
  float p0;                   // local pressure
  float vx;                   // x velocity component
  float vy;                   // y velocity component
  float v0;                   // velocity combination
  FILE* fp;                   // file pointer

  if (parameters.rank == MASTER)
  {
    // open the final state file
    fp = fopen(FSFILE, "w");

    // write the final state values
    for (int jj = 0; jj < parameters.gy; jj++)
    {
      for (int ii = 0; ii < parameters.gx; ii++)
      {
        if (global_obstacles[ii + jj * parameters.gx])
        {
          vx = vy = v0 = 0.f;
          p0 = parameters.density * c0;
        }
        else
        {
          l0 = global_grid->s0[ii + jj * parameters.gx] + global_grid->s1[ii + jj * parameters.gx] + global_grid->s2[ii + jj * parameters.gx] + global_grid->s3[ii + jj * parameters.gx] + global_grid->s4[ii + jj * parameters.gx] + global_grid->s5[ii + jj * parameters.gx] + global_grid->s6[ii + jj * parameters.gx] + global_grid->s7[ii + jj * parameters.gx] + global_grid->s8[ii + jj * parameters.gx];
          vx = (global_grid->s1[ii + jj * parameters.gx] + global_grid->s5[ii + jj * parameters.gx] + global_grid->s8[ii + jj * parameters.gx] - (global_grid->s3[ii + jj * parameters.gx] + global_grid->s6[ii + jj * parameters.gx] + global_grid->s7[ii + jj * parameters.gx])) / l0;
          vy = (global_grid->s2[ii + jj * parameters.gx] + global_grid->s5[ii + jj * parameters.gx] + global_grid->s6[ii + jj * parameters.gx] - (global_grid->s4[ii + jj * parameters.gx] + global_grid->s7[ii + jj * parameters.gx] + global_grid->s8[ii + jj * parameters.gx])) / l0;
          v0 = sqrtf((vx * vx) + (vy * vy));
          p0 = l0 * c0;
        }

        fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, vx, vy, v0, p0, global_obstacles[ii + jj * parameters.gx]);
      }
    }

    // close the final state file
    fclose(fp);

    // open the average velocity file
    fp = fopen(AVFILE, "w");

    // write the average velocity values
    for (int tt = 0; tt < parameters.iterations; tt++)
    {
      fprintf(fp, "%d:\t%.12E\n", tt, global_average_velocity[tt]);
    }

    // close the average velocity file
    fclose(fp);
  }

  return EXIT_SUCCESS;
}

// finalise - free allocated memory
int finalise(parameters* parameters_ptr, grid** global_grid_ptr, grid** local_old_grid_ptr, grid** local_new_grid_ptr, float** halo_snd_grid_ptr, float** halo_rcv_grid_ptr, char** global_obstacles_ptr, char** local_obstacles_ptr, float** global_average_velocity_ptr, float** local_total_velocity_ptr)
{
  if (parameters_ptr->rank == MASTER)
  {
    _mm_free((*global_grid_ptr)->s0);
    (*global_grid_ptr)->s0 = NULL;
    _mm_free((*global_grid_ptr)->s1);
    (*global_grid_ptr)->s1 = NULL;
    _mm_free((*global_grid_ptr)->s2);
    (*global_grid_ptr)->s2 = NULL;
    _mm_free((*global_grid_ptr)->s3);
    (*global_grid_ptr)->s3 = NULL;
    _mm_free((*global_grid_ptr)->s4);
    (*global_grid_ptr)->s4 = NULL;
    _mm_free((*global_grid_ptr)->s5);
    (*global_grid_ptr)->s5 = NULL;
    _mm_free((*global_grid_ptr)->s6);
    (*global_grid_ptr)->s6 = NULL;
    _mm_free((*global_grid_ptr)->s7);
    (*global_grid_ptr)->s7 = NULL;
    _mm_free((*global_grid_ptr)->s8);
    (*global_grid_ptr)->s8 = NULL;
    _mm_free(*global_grid_ptr);
    *global_grid_ptr = NULL;
  }

  _mm_free((*local_old_grid_ptr)->s0);
  (*local_old_grid_ptr)->s0 = NULL;
  _mm_free((*local_old_grid_ptr)->s1);
  (*local_old_grid_ptr)->s1 = NULL;
  _mm_free((*local_old_grid_ptr)->s2);
  (*local_old_grid_ptr)->s2 = NULL;
  _mm_free((*local_old_grid_ptr)->s3);
  (*local_old_grid_ptr)->s3 = NULL;
  _mm_free((*local_old_grid_ptr)->s4);
  (*local_old_grid_ptr)->s4 = NULL;
  _mm_free((*local_old_grid_ptr)->s5);
  (*local_old_grid_ptr)->s5 = NULL;
  _mm_free((*local_old_grid_ptr)->s6);
  (*local_old_grid_ptr)->s6 = NULL;
  _mm_free((*local_old_grid_ptr)->s7);
  (*local_old_grid_ptr)->s7 = NULL;
  _mm_free((*local_old_grid_ptr)->s8);
  (*local_old_grid_ptr)->s8 = NULL;
  _mm_free(*local_old_grid_ptr);
  *local_old_grid_ptr = NULL;

  _mm_free((*local_new_grid_ptr)->s0);
  (*local_new_grid_ptr)->s0 = NULL;
  _mm_free((*local_new_grid_ptr)->s1);
  (*local_new_grid_ptr)->s1 = NULL;
  _mm_free((*local_new_grid_ptr)->s2);
  (*local_new_grid_ptr)->s2 = NULL;
  _mm_free((*local_new_grid_ptr)->s3);
  (*local_new_grid_ptr)->s3 = NULL;
  _mm_free((*local_new_grid_ptr)->s4);
  (*local_new_grid_ptr)->s4 = NULL;
  _mm_free((*local_new_grid_ptr)->s5);
  (*local_new_grid_ptr)->s5 = NULL;
  _mm_free((*local_new_grid_ptr)->s6);
  (*local_new_grid_ptr)->s6 = NULL;
  _mm_free((*local_new_grid_ptr)->s7);
  (*local_new_grid_ptr)->s7 = NULL;
  _mm_free((*local_new_grid_ptr)->s8);
  (*local_new_grid_ptr)->s8 = NULL;
  _mm_free(*local_new_grid_ptr);
  *local_new_grid_ptr = NULL;

  _mm_free(*halo_snd_grid_ptr);
  *halo_snd_grid_ptr = NULL;

  _mm_free(*halo_rcv_grid_ptr);
  *halo_rcv_grid_ptr = NULL;

  if (parameters_ptr->rank == MASTER)
  {
  _mm_free(*global_obstacles_ptr);
  *global_obstacles_ptr = NULL;
  }

  _mm_free(*local_obstacles_ptr);
  *local_obstacles_ptr = NULL;

  if (parameters_ptr->rank == MASTER)
  {
  _mm_free(*global_average_velocity_ptr);
  *global_average_velocity_ptr = NULL;
  }

  _mm_free(*local_total_velocity_ptr);
  *local_total_velocity_ptr = NULL;

  return EXIT_SUCCESS;
}