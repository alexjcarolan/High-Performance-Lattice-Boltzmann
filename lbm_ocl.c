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

#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS 9
#define OCLFILE "lbm_ocl.cl"
#define FSFILE  "outputs/final_state.data"
#define AVFILE  "outputs/average_velocity.data"

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

// structure to hold objects
typedef struct
{
  cl_device_id     device;   // device 
  cl_context       context;  // context
  cl_command_queue queue;    // queue

  cl_program program;        // program
  cl_kernel  initiate;       // initiate
  cl_kernel  simulate;       // simulate

  cl_mem old_grid;           // old grid
  cl_mem new_grid;           // new grid
  cl_mem tmp_grid;           // tmp grid
  cl_mem obstacles;          // obstacles
  cl_mem iteration_velocity; // iteration velocity
} objects;

// function prototypes
float realise(const parameters parameters, objects objects, float* restrict old_grid, float* restrict new_grid, const char* restrict obstacles, float* restrict iteration_velocity, int const total_cells);
int initialise(parameters* parameters_ptr, objects* objects_ptr, float** old_grid_ptr, float** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr, float** iteration_velocity_ptr, int* total_cells_ptr, const char* parameters_file, const char* obstacles_file);
int serialise(const parameters parameters, const float* old_grid, const char* obstacles, const float* average_velocity);
int finalise(parameters* parameters_ptr, objects* objects_ptr, float** old_grid_ptr, float** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr, float** iteration_velocity_ptr);

// main - initialise, realise, serialise, finalise
int main(int argc, char* argv[])
{
  parameters parameters;            // simulation parameters
  objects    objects;               // heterogeneous objects
  float* old_grid           = NULL; // old grid of cells
  float* new_grid           = NULL; // new grid of cells
  char*  obstacles          = NULL; // grid of obstacles
  float* average_velocity   = NULL; // average velocity
  float* iteration_velocity = NULL; // iteration velocity
  int    total_cells;               // total cells
  float  reynolds_number;           // reynolds number
  char*  parameters_file    = NULL; // name of parameters file
  char*  obstacles_file     = NULL; // name of obstacles file
  struct timeval time;              // structure to hold elapsed time
  double tic, toc;                  // float to record elapsed time

  // parse command line arguments
  parameters_file = argv[1];
  obstacles_file = argv[2];

  // load data structures
  initialise(&parameters, &objects, &old_grid, &new_grid, &obstacles, &average_velocity, &iteration_velocity, &total_cells, parameters_file, obstacles_file);

  // get initial time
  gettimeofday(&time, NULL);
  tic = time.tv_sec + (time.tv_usec / 1000000.0);

  // iterate timestep
  for (int tt = 0; tt < parameters.iterations; tt++)
  {
    average_velocity[tt] = realise(parameters, objects, old_grid, new_grid, obstacles, iteration_velocity, total_cells); 
    objects.tmp_grid = objects.old_grid;
    objects.old_grid = objects.new_grid;
    objects.new_grid = objects.tmp_grid;
  }
  clEnqueueReadBuffer(objects.queue, objects.old_grid, CL_TRUE, 0, sizeof(float) * parameters.nx * parameters.ny * NSPEEDS, old_grid, 0, NULL, NULL);

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
  finalise(&parameters, &objects, &old_grid, &new_grid, &obstacles, &average_velocity, &iteration_velocity);

  return EXIT_SUCCESS;
}

// realise - accelerate, propogate, collide, rebound
float realise(const parameters parameters, objects objects, float* restrict old_grid, float* restrict new_grid, const char* restrict obstacles, float* restrict iteration_velocity, int const total_cells)
{
  size_t global[2] = {parameters.nx, parameters.ny}; // global indices

  // set arguments
  clSetKernelArg(objects.initiate, 1, sizeof(cl_mem), &objects.old_grid);
  clSetKernelArg(objects.simulate, 1, sizeof(cl_mem), &objects.old_grid);
  clSetKernelArg(objects.simulate, 2, sizeof(cl_mem), &objects.new_grid);

  // Queue kernels
  clEnqueueNDRangeKernel(objects.queue, objects.initiate, 1, NULL, global, NULL, 0, NULL, NULL);
  clEnqueueNDRangeKernel(objects.queue, objects.simulate, 2, NULL, global, NULL, 0, NULL, NULL);

  // Read iteration velocity
  clEnqueueReadBuffer(objects.queue, objects.iteration_velocity, CL_TRUE, 0, sizeof(float) * parameters.nx * parameters.ny, iteration_velocity, 0, NULL, NULL);

  // compute average velocity
  float total_velocity = 0.f;
  for (int jj = 0; jj < parameters.ny; jj++)
  {
    for (int ii = 0; ii < parameters.nx; ii++)
    {
      total_velocity += iteration_velocity[ii + jj * parameters.nx];
    }
  }

  return total_velocity / total_cells;
}

// initialise - open, allocate, initialise, close
int initialise(parameters* parameters_ptr, objects* objects_ptr, float** old_grid_ptr, float** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr, float** iteration_velocity_ptr, int* total_cells_ptr, const char* parameters_file, const char* obstacles_file)
{
  cl_platform_id platforms[8]; // platform list
  cl_device_id   devices[32];  // device list
  char  message[1024];         // message buffer
  int   occupied;              // cell occupation
  int   ii, jj;                // array indices
  char* source;                // kernel source
  long  size;                  // kernel size 
  FILE* fp;                    // file pointer

  // open the parameters file
  fp = fopen(parameters_file, "r");

  // read the parameters values
  fscanf(fp, "%d\n", &(parameters_ptr->nx));
  fscanf(fp, "%d\n", &(parameters_ptr->ny));
  fscanf(fp, "%d\n", &(parameters_ptr->reynolds));
  fscanf(fp, "%d\n", &(parameters_ptr->iterations));
  fscanf(fp, "%f\n", &(parameters_ptr->density));
  fscanf(fp, "%f\n", &(parameters_ptr->relaxation));
  fscanf(fp, "%f\n", &(parameters_ptr->acceleration));

  // close the parameters file
  fclose(fp);

  // allocate old grid
  *old_grid_ptr = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny * NSPEEDS, 64);
  
  // allocate new grid
  *new_grid_ptr = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny * NSPEEDS, 64);
  
  // allocate obstacles
  *obstacles_ptr = _mm_malloc(sizeof(char) * parameters_ptr->nx * parameters_ptr->ny, 64);

  // initialise weighting factors
  const float w0 = parameters_ptr->density * 4.f / 9.f;
  const float w1 = parameters_ptr->density       / 9.f;
  const float w2 = parameters_ptr->density       / 36.f;

  // initialise grids
  for (int jj = 0; jj < parameters_ptr->ny; jj++)
  {
    for (int ii = 0; ii < parameters_ptr->nx; ii++)
    {
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 0 * parameters_ptr->nx * parameters_ptr->ny] = w0;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 1 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 2 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 3 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 4 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 5 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 6 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 7 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*old_grid_ptr)[ii + jj * parameters_ptr->nx + 8 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 0 * parameters_ptr->nx * parameters_ptr->ny] = w0;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 1 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 2 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 3 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 4 * parameters_ptr->nx * parameters_ptr->ny] = w1;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 5 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 6 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 7 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*new_grid_ptr)[ii + jj * parameters_ptr->nx + 8 * parameters_ptr->nx * parameters_ptr->ny] = w2;
      (*obstacles_ptr)[ii + jj * parameters_ptr->nx] = 0;
    }
  }

  // open the obstacles file
  fp = fopen(obstacles_file, "r");

  // read the obstacles values
  while ((fscanf(fp, "%d %d %d\n", &ii, &jj, &occupied)) != EOF)
  {
    (*obstacles_ptr)[ii + jj * parameters_ptr->nx] = occupied;
  }

  // close the obstacles file
  fclose(fp);

  // allocate average velocity
  *average_velocity_ptr = _mm_malloc(sizeof(float) * parameters_ptr->iterations, 64);

  // allocate iteration velocity
  *iteration_velocity_ptr = _mm_malloc(sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, 64);

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

  // open the kernel file
  fp = fopen(OCLFILE, "r");

  // Load kernel source
  fseek(fp, 0, SEEK_END);
  size = ftell(fp) + 1;
  source = malloc(size);
  memset(source, 0, size);
  fseek(fp, 0, SEEK_SET);
  fread(source, 1, size, fp);

  // close the kernel file
  fclose(fp);

  // Get platforms
  clGetPlatformIDs(8, platforms, NULL);

  // Get devices
  clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 32, devices, NULL);

  // Create device
  objects_ptr->device = devices[0];

  // Create context
  objects_ptr->context = clCreateContext(NULL, 1, &objects_ptr->device, NULL, NULL, NULL);

  // Create queue
  objects_ptr->queue = clCreateCommandQueue(objects_ptr->context, objects_ptr->device, 0, NULL);

  // Create program
  objects_ptr->program = clCreateProgramWithSource(objects_ptr->context, 1, (const char**)&source, NULL, NULL);

  // Build program
  clBuildProgram(objects_ptr->program, 1, &objects_ptr->device, "", NULL, NULL);

  // Create kernels
  objects_ptr->initiate = clCreateKernel(objects_ptr->program, "initiate", NULL);
  objects_ptr->simulate = clCreateKernel(objects_ptr->program, "simulate", NULL);

  // Allocate buffers
  objects_ptr->old_grid = clCreateBuffer(objects_ptr->context, CL_MEM_READ_WRITE, sizeof(float) * parameters_ptr->nx * parameters_ptr->ny * NSPEEDS, NULL, NULL);
  objects_ptr->new_grid = clCreateBuffer(objects_ptr->context, CL_MEM_READ_WRITE, sizeof(float) * parameters_ptr->nx * parameters_ptr->ny * NSPEEDS, NULL, NULL);
  objects_ptr->obstacles = clCreateBuffer(objects_ptr->context, CL_MEM_READ_ONLY, sizeof(char) * parameters_ptr->nx * parameters_ptr->ny, NULL, NULL);
  objects_ptr->iteration_velocity = clCreateBuffer(objects_ptr->context, CL_MEM_READ_WRITE, sizeof(float) * parameters_ptr->nx * parameters_ptr->ny, NULL, NULL);

  // Initialise buffers
  clEnqueueWriteBuffer(objects_ptr->queue, objects_ptr->old_grid, CL_TRUE, 0, sizeof(float) * parameters_ptr->nx * parameters_ptr->ny * NSPEEDS, *old_grid_ptr, 0, NULL, NULL);
  clEnqueueWriteBuffer(objects_ptr->queue, objects_ptr->obstacles, CL_TRUE, 0, sizeof(char) * parameters_ptr->nx * parameters_ptr->ny, *obstacles_ptr, 0, NULL, NULL);

  // Initialise arguments
  clSetKernelArg(objects_ptr->initiate, 0, sizeof(parameters), parameters_ptr);
  clSetKernelArg(objects_ptr->initiate, 1, sizeof(cl_mem), &objects_ptr->old_grid);
  clSetKernelArg(objects_ptr->initiate, 2, sizeof(cl_mem), &objects_ptr->obstacles);
  clSetKernelArg(objects_ptr->simulate, 0, sizeof(parameters), parameters_ptr);
  clSetKernelArg(objects_ptr->simulate, 1, sizeof(cl_mem), &objects_ptr->old_grid);
  clSetKernelArg(objects_ptr->simulate, 2, sizeof(cl_mem), &objects_ptr->new_grid);
  clSetKernelArg(objects_ptr->simulate, 3, sizeof(cl_mem), &objects_ptr->obstacles);
  clSetKernelArg(objects_ptr->simulate, 4, sizeof(cl_mem), &objects_ptr->iteration_velocity);

  free(source);

  return EXIT_SUCCESS;
}

// write - final state, average velocity
int serialise(const parameters parameters, const float* old_grid, const char* obstacles, const float* average_velocity)
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
        l0 = old_grid[ii + jj * parameters.nx + 0 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 1 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 2 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 3 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 4 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 5 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 6 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 7 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 8 * parameters.nx * parameters.ny];
        vx = (old_grid[ii + jj * parameters.nx + 1 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 5 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 8 * parameters.nx * parameters.ny] - (old_grid[ii + jj * parameters.nx + 3 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 6 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 7 * parameters.nx * parameters.ny])) / l0;
        vy = (old_grid[ii + jj * parameters.nx + 2 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 5 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 6 * parameters.nx * parameters.ny] - (old_grid[ii + jj * parameters.nx + 4 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 7 * parameters.nx * parameters.ny] + old_grid[ii + jj * parameters.nx + 8 * parameters.nx * parameters.ny])) / l0;
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
int finalise(parameters* parameters_ptr, objects* objects_ptr, float** old_grid_ptr, float** new_grid_ptr, char** obstacles_ptr, float** average_velocity_ptr, float** iteration_velocity_ptr)
{
  _mm_free(*old_grid_ptr);
  *old_grid_ptr = NULL;

  _mm_free(*new_grid_ptr);
  *new_grid_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*average_velocity_ptr);
  *average_velocity_ptr = NULL;

  _mm_free(*iteration_velocity_ptr);
  *iteration_velocity_ptr = NULL;

  clReleaseMemObject(objects_ptr->old_grid);

  clReleaseMemObject(objects_ptr->new_grid);

  clReleaseMemObject(objects_ptr->obstacles);

  clReleaseMemObject(objects_ptr->iteration_velocity);

  clReleaseProgram(objects_ptr->program);

  clReleaseKernel(objects_ptr->initiate);

  clReleaseKernel(objects_ptr->simulate);

  clReleaseContext(objects_ptr->context);
  
  clReleaseCommandQueue(objects_ptr->queue);

  return EXIT_SUCCESS;
}