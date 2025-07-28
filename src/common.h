#ifndef PARTICLELIFE_COMMON_H
#define PARTICLELIFE_COMMON_H

struct Particle {
    float x, y;
    float vx, vy;
    int state;
	float energy;
	float potential;
};

struct Rule {
    float attraction;
    float range;
	float power;
};

__global__ void compute_forces(Particle* particles, Rule* rules, int num_particles, int num_states, float* fx, float* fy);
__global__ void integrate(Particle* particles, float* fx, float* fy, int num_particles, float dt, float max_velocity, float *target_energy, float *average_energy);
__global__ void update_states(Particle* particles, int num_particles, int num_states, float dt);
__global__ void get_avg(Particle* particles, int num_particles, float *average_energy);

extern "C" void launch_kernels(Particle* d_particles, Rule* d_rules, int num_particles, int num_states, float* d_fx, float* d_fy, float dt, float max_velocity, float *target_energy, float *average_energy);

#endif