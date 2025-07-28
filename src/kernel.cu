#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"

__global__ void compute_forces(
    Particle* particles,
    Rule* rules,
    int num_particles,
    int num_states,
    float* force_x,
    float* force_y
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle p_i = particles[i];
    float fx = 0.0f, fy = 0.0f;
	
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;
        Particle p_j = particles[j];
		Rule rule = rules[p_i.state * num_states + p_j.state];
        float dx = p_j.x - p_i.x;
        float dy = p_j.y - p_i.y;
        float dist2 = dx * dx + dy * dy;
        float dist = sqrtf(dist2 + 1e-6f);

        float influence = expf(-dist * dist / (2.0f * rule.range * rule.range));
        float force = (rule.attraction + rule.power * p_j.energy) * influence;

        fx += force * dx;
        fy += force * dy;
    }

    force_x[i] = fx;
    force_y[i] = fy;
}

__global__ void get_avg(Particle* particles, int num_particles, float *average_energy)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
	Particle& p = particles[i];
	atomicAdd(average_energy, p.energy);
}

__global__ void integrate(
    Particle* particles,
    float* force_x,
    float* force_y,
    int num_particles,
    float dt,
	float max_velocity,
	float* target_energy,
	float* average_energy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
	p.energy += (*target_energy * num_particles - *average_energy) / num_particles;
	if (sqrtf(pow(p.vx, 2.) + pow(p.vy, 2.)) < max_velocity) {
		p.vx += force_x[i] * dt;
		p.vy += force_y[i] * dt;
	} else {
		p.vx *= powf(1 - dt, 2);
		p.vy *= powf(1 - dt, 2);
	}
    p.x += p.vx * dt * p.energy;
    p.y += p.vy * dt * p.energy;
}

__global__ void update_states(
    Particle* particles,
    int num_particles,
    int num_states,
	float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
	if (p.potential > sqrtf(pow(p.vx, 2) + pow(p.vy, 2)) * num_states) {
		p.state = (p.state + (p.potential > p.energy ? 1 : -1) + num_states) % num_states;
		p.potential *= powf(1 - dt, 2);
		p.energy *= 1 - dt;
	}
	p.potential += p.energy - p.potential * dt;
}

extern "C" void launch_kernels(Particle* d_particles, Rule* d_rules, int num_particles, int num_states, float* d_fx, float* d_fy, float dt, float max_velocity, float *target_energy, float *average_energy) {
    int blockSize = 256;
    int gridSize = (num_particles + blockSize - 1) / blockSize;
	
    compute_forces<<<gridSize, blockSize>>>(d_particles, d_rules, num_particles, num_states, d_fx, d_fy);
    cudaDeviceSynchronize();
	
	get_avg<<<gridSize, blockSize>>>(d_particles, num_particles, average_energy);
    cudaDeviceSynchronize();

    integrate<<<gridSize, blockSize>>>(d_particles, d_fx, d_fy, num_particles, dt, max_velocity, target_energy, average_energy);
    cudaDeviceSynchronize();

    update_states<<<gridSize, blockSize>>>(d_particles, num_particles, num_states, dt);
    cudaDeviceSynchronize();
}