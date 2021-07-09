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

kernel void initiate(const parameters parameters, global float* restrict old_grid, global const char* restrict obstacles)
{
  int ii = get_global_id(0);

  const float a1 = parameters.density * parameters.acceleration / 9.f;  // acceleration factor
  const float a2 = parameters.density * parameters.acceleration / 36.f; // acceleration factor

  // accelerate flow of unoccupied cells
  if (!obstacles[ii + (parameters.ny - 2) * parameters.nx] && (old_grid[ii + (parameters.ny - 2) * parameters.nx + 3 * parameters.nx * parameters.ny] - a1) > 0.f && (old_grid[ii + (parameters.ny - 2) * parameters.nx + 6 * parameters.nx * parameters.ny] - a2) > 0.f && (old_grid[ii + (parameters.ny - 2) * parameters.nx + 7 * parameters.nx * parameters.ny] - a2) > 0.f)
  {
    // increase east-side speeds
    old_grid[ii + (parameters.ny - 2) * parameters.nx + 1 * parameters.nx * parameters.ny] += a1;
    old_grid[ii + (parameters.ny - 2) * parameters.nx + 5 * parameters.nx * parameters.ny] += a2;
    old_grid[ii + (parameters.ny - 2) * parameters.nx + 8 * parameters.nx * parameters.ny] += a2;
    // decrease west-side speeds
    old_grid[ii + (parameters.ny - 2) * parameters.nx + 3 * parameters.nx * parameters.ny] -= a1;
    old_grid[ii + (parameters.ny - 2) * parameters.nx + 6 * parameters.nx * parameters.ny] -= a2;
    old_grid[ii + (parameters.ny - 2) * parameters.nx + 7 * parameters.nx * parameters.ny] -= a2;
  }
}

kernel void simulate(const parameters parameters, global float* restrict old_grid, global float* restrict new_grid, global const char* obstacles, global float* restrict iteration_velocity)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  const float w0 = 4.f / 9.f;  // weighting factor
  const float w1 = 1.f / 9.f;  // weighting factor
  const float w2 = 1.f / 36.f; // weighting factor

  const int xe = (ii == parameters.nx - 1) ? (0) : (ii + 1);
  const int yn = (jj == parameters.ny - 1) ? (0) : (jj + 1);
  const int xw = (ii == 0) ? (ii + parameters.nx - 1) : (ii - 1);
  const int ys = (jj == 0) ? (jj + parameters.ny - 1) : (jj - 1);

  // propogate flow from neighbouring cells
  const float s0 = old_grid[ii + jj * parameters.nx + 0 * parameters.nx * parameters.ny]; // centre
  const float s1 = old_grid[xw + jj * parameters.nx + 1 * parameters.nx * parameters.ny]; // east
  const float s2 = old_grid[ii + ys * parameters.nx + 2 * parameters.nx * parameters.ny]; // north
  const float s3 = old_grid[xe + jj * parameters.nx + 3 * parameters.nx * parameters.ny]; // west
  const float s4 = old_grid[ii + yn * parameters.nx + 4 * parameters.nx * parameters.ny]; // south
  const float s5 = old_grid[xw + ys * parameters.nx + 5 * parameters.nx * parameters.ny]; // north-east
  const float s6 = old_grid[xe + ys * parameters.nx + 6 * parameters.nx * parameters.ny]; // north-west
  const float s7 = old_grid[xe + yn * parameters.nx + 7 * parameters.nx * parameters.ny]; // south-west
  const float s8 = old_grid[xw + yn * parameters.nx + 8 * parameters.nx * parameters.ny]; // south-east

  // rebound flow of occupied cells
  if (obstacles[ii + jj * parameters.nx])
  {
    new_grid[ii + jj * parameters.nx + 1 * parameters.nx * parameters.ny] = s3;
    new_grid[ii + jj * parameters.nx + 2 * parameters.nx * parameters.ny] = s4;
    new_grid[ii + jj * parameters.nx + 3 * parameters.nx * parameters.ny] = s1;
    new_grid[ii + jj * parameters.nx + 4 * parameters.nx * parameters.ny] = s2;
    new_grid[ii + jj * parameters.nx + 5 * parameters.nx * parameters.ny] = s7;
    new_grid[ii + jj * parameters.nx + 6 * parameters.nx * parameters.ny] = s8;
    new_grid[ii + jj * parameters.nx + 7 * parameters.nx * parameters.ny] = s5;
    new_grid[ii + jj * parameters.nx + 8 * parameters.nx * parameters.ny] = s6;
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
    new_grid[ii + jj * parameters.nx + 0 * parameters.nx * parameters.ny] = s0 + parameters.relaxation * (d0 - s0);
    new_grid[ii + jj * parameters.nx + 1 * parameters.nx * parameters.ny] = s1 + parameters.relaxation * (d1 - s1);
    new_grid[ii + jj * parameters.nx + 2 * parameters.nx * parameters.ny] = s2 + parameters.relaxation * (d2 - s2);
    new_grid[ii + jj * parameters.nx + 3 * parameters.nx * parameters.ny] = s3 + parameters.relaxation * (d3 - s3);
    new_grid[ii + jj * parameters.nx + 4 * parameters.nx * parameters.ny] = s4 + parameters.relaxation * (d4 - s4);
    new_grid[ii + jj * parameters.nx + 5 * parameters.nx * parameters.ny] = s5 + parameters.relaxation * (d5 - s5);
    new_grid[ii + jj * parameters.nx + 6 * parameters.nx * parameters.ny] = s6 + parameters.relaxation * (d6 - s6);
    new_grid[ii + jj * parameters.nx + 7 * parameters.nx * parameters.ny] = s7 + parameters.relaxation * (d7 - s7);
    new_grid[ii + jj * parameters.nx + 8 * parameters.nx * parameters.ny] = s8 + parameters.relaxation * (d8 - s8);

    iteration_velocity[ii + jj * parameters.nx] = sqrt(v0);
  }
}