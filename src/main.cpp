#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "common.h"

#define BLOCK_SIZE 256

extern "C" void launch_kernels(Particle* d_particles, Rule* d_rules, int num_particles, int num_states, float* d_fx, float* d_fy, int width, int height, float dt, float max_velocity, float *target_energy, float *average_energy);

namespace py = pybind11;

static float h_average_energy = 0;

void step_simulation(py::array_t<Particle> particles_np, py::array_t<Rule> rules_np, float dt, int width, int height, float max_velocity, float target_energy) {
    auto buf_particles = particles_np.request();
    auto buf_rules = rules_np.request();

    int num_particles = buf_particles.size;
    int num_states = (int)std::sqrt(buf_rules.size); // assumes square matrix

    if (buf_particles.ndim != 1 || buf_rules.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1D");
    }

    // Host pointers
    Particle* h_particles = static_cast<Particle*>(buf_particles.ptr);
    Rule* h_rules = static_cast<Rule*>(buf_rules.ptr);

    // Allocate device memory
    Particle* d_particles;
    Rule* d_rules;
    float* d_fx;
    float* d_fy;
	float* average_energy;
	float* d_target_energy;
	cudaMalloc((void**)&average_energy, sizeof(float));
	cudaMalloc((void**)&d_target_energy, sizeof(float));

    cudaMalloc(&d_particles, sizeof(Particle) * num_particles);
    cudaMalloc(&d_rules, sizeof(Rule) * num_states * num_states);
    cudaMalloc(&d_fx, sizeof(float) * num_particles);
    cudaMalloc(&d_fy, sizeof(float) * num_particles);

    // Copy to device
    cudaMemcpy(d_particles, h_particles, sizeof(Particle) * num_particles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rules, h_rules, sizeof(Rule) * num_states * num_states, cudaMemcpyHostToDevice);
	cudaMemcpy(d_target_energy, &target_energy, sizeof(float), cudaMemcpyHostToDevice);
    // Run CUDA kernels
    launch_kernels(d_particles, d_rules, num_particles, num_states, d_fx, d_fy, width, height, dt, max_velocity, d_target_energy, average_energy);

    // Copy result back to host
    cudaMemcpy(h_particles, d_particles, sizeof(Particle) * num_particles, cudaMemcpyDeviceToHost);

    // Cleanup
	cudaFree(average_energy);
	cudaFree(d_target_energy);
    cudaFree(d_particles);
    cudaFree(d_rules);
    cudaFree(d_fx);
    cudaFree(d_fy);
}

PYBIND11_MODULE(particlelife_cuda, m) {

	PYBIND11_NUMPY_DTYPE(Particle, x, y, vx, vy, state, energy, potential);
	PYBIND11_NUMPY_DTYPE(Rule, attraction, range, power);

    m.def("step_simulation", &step_simulation,
          py::arg("particles"), py::arg("rules"), py::arg("dt"), py::arg("width"), py::arg("height"), py::arg("max_velocity"), py::arg("target_energy"));
}
