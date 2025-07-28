#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"



__global__ void update_states(
    Particle* particles,
    int num_particles,
    int num_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
    // Example: Change state based on x-coordinate
    if (p.x < 0.3f) p.state = (p.state + 1) % num_states;
}