#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include "common.h"

#define BLOCK_SIZE 256
#define SMOOTH_BLOCK_SIZE 24
#define MAX_STATES 32
#define PI 3.141592653589f
#define PARTICLE_COLLISION_RADIUS 1.0f

__global__ void compute_forces(
    Particle* particles,
    Rule* rules,
    int num_particles,
    int num_states,
	float dt,
    float* force_x,
    float* force_y,
	float* mass
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p_i = particles[i];
    float fx = 0.0f, fy = 0.0f;
	
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue;
        Particle p_j = particles[j];
		Rule rule = rules[p_i.state * MAX_STATES + p_j.state];
        float dx = p_j.x - p_i.x;
        float dy = p_j.y - p_i.y;
        float dist2 = dx * dx + dy * dy;
        float dist = sqrtf(dist2 + 1e-6f);
		float influence = expf(-dist * dist / (rule.range * rule.range * 0.5f)) * p_j.energy;
		float force = (((mass[p_i.state] * rule.power + dist2) * (mass[p_j.state] * rule.power + dist2)) / (mass[p_i.state] * mass[p_j.state] * rule.power * rule.power)) * rule.attraction * influence;

		fx += dx * force / dist;
		fy += dy * force / dist;
    }
    force_x[i] = fx / mass[p_i.state] * p_i.energy;
    force_y[i] = fy / mass[p_i.state] * p_i.energy;
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
	float* mass
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
	p.vx += force_x[i] * dt;
	p.vy += force_y[i] * dt;
	float speed = sqrtf(p.vx * p.vx + p.vy * p.vy);
	float max_speed = max_velocity;
	if (speed > max_speed) {
		float limit_scale = max_speed / speed;
		p.vx *= limit_scale;
		p.vy *= limit_scale;
	}
    p.x += p.vx * dt;
    p.y += p.vy * dt;
}

__global__ void update_states(
    Particle* particles,
    int num_particles,
    int num_states,
	float dt,
	float* mass,
	float* average_energy,
	float* target_energy,
	float* potential_gain,
	float* d_energy_mul,
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
	if (p.potential > (p.energy * p.energy)) {
		int old_state = p.state;
		p.state = (p.state + 1) % num_states;
		p.energy *= mass[p.state] / mass[old_state];
		p.potential /= mass[p.state] / mass[old_state];
	}
	float energy_adjustment = mass[p.state] * *target_energy;
	p.energy = p.energy * (1 - dt) + dt * energy_adjustment;
	p.energy = fmaxf(p.energy, 1e-4f);
	p.potential += dt * *potential_gain / p.energy;
}

__global__ void update_collisions(
    Particle* particles,
    int num_particles,
	int* particle_size,
	float* mass,
	float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
	
    Particle& p_i = particles[i];
	for (int j = 0; j < num_particles; ++j) {
		Particle& p_j = particles[j];
		float dx = p_i.x - p_j.x;
        float dy = p_i.y - p_j.y;
        float dist2 = dx * dx + dy * dy + 1e-6f;
		if (sqrtf(dist2) < PARTICLE_COLLISION_RADIUS) {
			float nx = dx / sqrtf(dist2);
			float ny = dy / sqrtf(dist2);
			float rvx = p_i.vx - p_j.vx;
			float rvy = p_i.vy - p_j.vy;
			float dp = rvx * nx + rvy * ny;
			if (dp > 0) continue;
			float im = -2.0f * dp / (1 / mass[p_i.state] + 1 / mass[p_j.state]);
			p_i.vx += im * nx / mass[p_i.state];
			p_i.vy += im * ny / mass[p_i.state];
			p_j.vx -= im * nx / mass[p_j.state];
			p_j.vy -= im * ny / mass[p_j.state];
			p_i.x += nx * (PARTICLE_COLLISION_RADIUS - sqrtf(dist2));
			p_i.y += ny * (PARTICLE_COLLISION_RADIUS - sqrtf(dist2));
			p_j.x -= nx * (PARTICLE_COLLISION_RADIUS - sqrtf(dist2));
			p_j.y -= ny * (PARTICLE_COLLISION_RADIUS - sqrtf(dist2));
			
		}
	}
}
	

__device__ __host__ inline void unpackRGBA(uint32_t packed, float& r, float& g, float& b, float& a) {
    r = float((packed) & 0xFF);
    g = float((packed >> 8) & 0xFF);
    b = float((packed >> 16) & 0xFF);
    a = float((packed >> 24) & 0xFF);
}

__device__ __host__ inline uint32_t packRGBA(float r, float g, float b, float a) {
    uint32_t R = min(max(int(a + 0.5f), 0), 255);
    uint32_t G = min(max(int(b + 0.5f), 0), 255);
    uint32_t B = min(max(int(g + 0.5f), 0), 255);
    uint32_t A = min(max(int(r + 0.5f), 0), 255);
    return (R << 24) | (G << 16) | (B << 8) | A;
}
static Particle* d_particles = nullptr;
static Particle* d_visible = nullptr;
static Rule* d_rules = nullptr;
static float* d_fx = nullptr;
static float* d_fy = nullptr;
static float* d_average_energy = nullptr;
static float* d_target_energy = nullptr;
static int* d_particle_size = nullptr;
static float* d_energy_mul = nullptr;
static float* d_mass = nullptr;
static float* d_potential_gain = nullptr;
static uint32_t* d_input = nullptr;
static uint32_t* d_output = nullptr;
static int* d_num_states = nullptr;
void init_cuda_managed_buffers(int num_particles, int num_states, int width, int height) {
    if (!d_particles)
        cudaMallocManaged(&d_particles, sizeof(Particle) * num_particles);
	if (!d_visible)
        cudaMallocManaged(&d_visible, sizeof(Particle) * num_particles);
    if (!d_rules)
        cudaMallocManaged(&d_rules, sizeof(Rule) * MAX_STATES * MAX_STATES);
    if (!d_fx)
        cudaMallocManaged(&d_fx, sizeof(float) * num_particles);
    if (!d_fy)
        cudaMallocManaged(&d_fy, sizeof(float) * num_particles);
    if (!d_average_energy)
        cudaMallocManaged(&d_average_energy, sizeof(float));
    if (!d_target_energy)
        cudaMallocManaged(&d_target_energy, sizeof(float));
	if (!d_mass)
		cudaMallocManaged(&d_mass, sizeof(float) * MAX_STATES);
	if (!d_energy_mul)
		cudaMallocManaged(&d_energy_mul, sizeof(float) * MAX_STATES);
	if (!d_input) {
		cudaMallocManaged(&d_input, width * height * sizeof(uint32_t));
	}
	if (!d_output) {
		cudaMallocManaged(&d_output, width * height * sizeof(uint32_t));
	}
	if (!d_potential_gain) {
		cudaMallocManaged(&d_potential_gain, sizeof(float));
	}
	if (!d_num_states) {
		cudaMallocManaged(&d_num_states, sizeof(int));
	}
	if (!d_particle_size) {
		cudaMallocManaged(&d_particle_size, sizeof(int));
	}
}

void free_cuda_managed_buffers() {
    if (d_particles) cudaFree(d_particles);
	if (d_visible) cudaFree(d_visible);
    if (d_rules) cudaFree(d_rules);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_average_energy) cudaFree(d_average_energy);
    if (d_target_energy) cudaFree(d_target_energy);
	if (d_mass) cudaFree(d_mass);
	if (d_energy_mul) cudaFree(d_energy_mul);
	if (d_input) cudaFree(d_input);
	if (d_potential_gain) cudaFree(d_potential_gain);
	if (d_num_states) cudaFree(d_num_states);
	if (d_particle_size) cudaFree(d_particle_size);
	
    d_particles = nullptr;
	d_visible = nullptr;
    d_rules = nullptr;
    d_fx = nullptr;
    d_fy = nullptr;
    d_average_energy = nullptr;
    d_target_energy = nullptr;
	d_energy_mul = nullptr;
	d_mass = nullptr;
	d_input = nullptr;
	d_potential_gain = nullptr;
	d_num_states = nullptr;
	d_particle_size = nullptr;
}

void step_simulation(int num_particles, int num_states, float dt, float max_velocity, float& target_energy, float& potential_gain, bool collide, int particle_size) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (num_particles + blockSize - 1) / blockSize;

    // Direct assignment instead of cudaMemcpy
    *d_target_energy = target_energy;
    *d_potential_gain = potential_gain;
	*d_particle_size = particle_size;
	*d_average_energy = 0.0f;
	cudaDeviceSynchronize();
	
	if (collide) {
		update_collisions<<<gridSize, blockSize>>>(d_particles, num_particles, d_particle_size, d_mass, dt);
		cudaDeviceSynchronize();
	}
	
	compute_forces<<<gridSize, blockSize>>>(
		d_particles, d_rules, num_particles, num_states, dt, 
		d_fx, d_fy, d_mass);
	cudaDeviceSynchronize();

	get_avg<<<gridSize, blockSize>>>(d_particles, num_particles, d_average_energy);
	cudaDeviceSynchronize();

	integrate<<<gridSize, blockSize>>>(d_particles, d_fx, d_fy, num_particles, dt, max_velocity, d_mass);
	cudaDeviceSynchronize();

	update_states<<<gridSize, blockSize>>>(
		d_particles, num_particles, num_states, dt, d_mass, 
		d_average_energy, d_target_energy, d_potential_gain, d_energy_mul);
	cudaDeviceSynchronize();
}
__global__ void render_voronoi(uint32_t* framebuffer, Particle* particles, int num_particles,
                                   int width, int height, int num_states, float offsetX, float offsetY, float zoom, float particle_size, float* mass, float* target_energy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float3 color = make_float3(0, 0, 0);
    float2 pixel = make_float2(x + 0.5f, y + 0.5f);
	float mindist2 = 1e20f;
	for (int i = 0; i < num_particles; ++i) {
		Particle p = particles[i];
        float px = ((p.x - offsetX - width / 2) * zoom + width / 2);
		float py = ((p.y - offsetY - height / 2) * zoom + height / 2);
        float dx = px - x;
        float dy = py - y;
        float dist2 = dx * dx + dy * dy;

        if (dist2 < mindist2) {
            mindist2 = dist2;
        }
    }
    for (int i = 0; i < num_particles; ++i) {
		Particle p = particles[i];
		float px = (p.x - offsetX - width / 2) * zoom + width / 2;
		float py = (p.y - offsetY - height / 2) * zoom + height / 2;
		if (px >= 0 && px < width && py >= 0 && py < height) {
			float2 pos = make_float2(px, py);
			float dx = px - x;
			float dy = py - y;
			float dist2 = dx * dx + dy * dy;
			float intensity = fmaxf(0.0f, fminf(1.0f, 1.0f - sqrtf(dist2 - mindist2) * particle_size));
			// Convert particle state to HSV color
			float h = p.state / (float)num_states;
			float s = fminf(1.0f, p.energy / mass[p.state] / *target_energy / 2) * 3 / 4 + 0.25;
			float v = intensity * (fminf(1.0f, fmaxf(0.0f, 1.0f - p.potential / (p.energy * p.energy))) * 3 / 4 + 0.25);

			// HSV → RGB
			float c = v * s;
			float h6 = h * 6.0f;
			float xcol = c * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
			float3 rgb;
			if      (h6 < 1) rgb = make_float3(c, xcol, 0);
			else if (h6 < 2) rgb = make_float3(xcol, c, 0);
			else if (h6 < 3) rgb = make_float3(0, c, xcol);
			else if (h6 < 4) rgb = make_float3(0, xcol, c);
			else if (h6 < 5) rgb = make_float3(xcol, 0, c);
			else             rgb = make_float3(c, 0, xcol);
			rgb.x += v - c;
			rgb.y += v - c;
			rgb.z += v - c;
			color.x += rgb.x;
			color.y += rgb.y;
			color.z += rgb.z;
		}
    }

    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);

    framebuffer[y * width + x] = packRGBA(
        (uint8_t)(color.x * 255.0f),
        (uint8_t)(color.y * 255.0f),
        (uint8_t)(color.z * 255.0f),
        255);
}

__device__ float gaussian_kernel(float u) {
	return (1.0 / sqrtf(2.0 * PI)) * expf(-0.5 * u * u);
}

__global__ void render_voronoi2(uint32_t* framebuffer, Particle* particles, int num_particles,
                                   int width, int height, int num_states, float offsetX, float offsetY, float zoom, float particle_size, float* mass, float* target_energy, bool flat, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float3 color = make_float3(0, 0, 0);
    float2 pixel = make_float2(x + 0.5f, y + 0.5f);
	float intensity_sum = 0.0f;
    for (int i = 0; i < num_particles; ++i) {
		Particle p = particles[i];
		float px = (p.x - offsetX - width / 2) * zoom + width / 2;
		float py = (p.y - offsetY - height / 2) * zoom + height / 2;
		if (px >= 0 && px < width && py >= 0 && py < height) {
			float dx = px - x;
			float dy = py - y;
			float dist2 = dx * dx + dy * dy;
			float intensity = 1.0f;
			if (!flat) {
				intensity = gaussian_kernel(sqrtf(dist2) / particle_size) / particle_size;
			} else {
				intensity = gaussian_kernel(sqrtf(dist2) / particle_size) / particle_size;
				intensity_sum += intensity;
				intensity = (intensity * 2) > threshold ? 1.0f : 0.0f;
			}
			
			// Convert particle state to HSV color
			float h = p.state / (float)num_states;
			float s = fminf(1.0f, p.energy / mass[p.state] / *target_energy / 2) * 3 / 4 + 0.25;
			float v = 1.0f;
			
			v = intensity * (fminf(1.0f, fmaxf(0.0f, 1.0f - p.potential / (p.energy * p.energy))) * 3 / 4 + 0.25);
			// HSV → RGB
			float c = v * s;
			float h6 = h * 6.0f;
			float xcol = c * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
			float3 rgb;
			if      (h6 < 1) rgb = make_float3(c, xcol, 0);
			else if (h6 < 2) rgb = make_float3(xcol, c, 0);
			else if (h6 < 3) rgb = make_float3(0, c, xcol);
			else if (h6 < 4) rgb = make_float3(0, xcol, c);
			else if (h6 < 5) rgb = make_float3(xcol, 0, c);
			else             rgb = make_float3(c, 0, xcol);
			rgb.x += v - c;
			rgb.y += v - c;
			rgb.z += v - c;
			if (flat) {
				color.x = color.x * (1 - intensity) + rgb.x * intensity;
				color.y = color.y * (1 - intensity) + rgb.y * intensity;
				color.z = color.z * (1 - intensity) + rgb.z * intensity;
			} else {
				color.x += rgb.x;
				color.y += rgb.y;
				color.z += rgb.z;
			}
		}
    }
	color.x = fminf(1.0f, color.x);
	color.z = fminf(1.0f, color.z);
	color.y = fminf(1.0f, color.y);
	if (!flat || intensity_sum > threshold) {
		framebuffer[y * width + x] = packRGBA(
			(uint8_t)(color.x * 255.0f),
			(uint8_t)(color.y * 255.0f),
			(uint8_t)(color.z * 255.0f),
			255);
	}
}

__global__ void add_trail(uint32_t* input, uint32_t* output, float dt,int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
	float r = 0, g = 0, b = 0, a = 0;
	unpackRGBA(output[y * width + x], r, g, b, a);
	float r2 = r * (1 - dt * 10) + (float)(input[y * width + x] & 255);
	float g2 = g * (1 - dt * 10) + (float)(input[y * width + x] >> 8 & 255);
	float b2 = b * (1 - dt * 10) + (float)(input[y * width + x] >> 16 & 255);
	output[y * width + x] = (255 << 24) | ((uint8_t)fmaxf(0.0f, fminf(255.0f, b2)) << 16) | ((uint8_t)fmaxf(0.0f, fminf(255.0f, g2)) << 8) | (uint8_t)fmaxf(0.0f, fminf(255.0f, r2));
}

int main(int argc, char* argv[]) {
	
    // Initialize SDL
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << "\n";
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Vector Display", width, height, 0);
    if (!window) {
        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        std::cerr << "SDL_CreateRenderer Error: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!texture) {
        std::cerr << "SDL_CreateTexture Error: " << SDL_GetError() << "\n";
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
	
    // Create host-side particle vector and rules vector
    std::vector<Particle> particles(num_particles);
    std::vector<Rule> rules(MAX_STATES * MAX_STATES);
	std::vector<float> mass(MAX_STATES);
	std::vector<float> energy_mul(MAX_STATES);

    // Initialize particles with random positions, velocities, states, and energy
	srand(time(0));
    for (Particle& p : particles) {
		p.x = ((float)rand() / RAND_MAX * width - width / 2) / zoom + width / 2 + offsetX;
		p.y = ((float)rand() / RAND_MAX * height - height / 2) / zoom + height / 2 + offsetY;
		p.vx = ((float)rand() / RAND_MAX - 0.5f);
		p.vy = ((float)rand() / RAND_MAX - 0.5f);
		p.state = rand() % num_states;
		p.energy = 1.0;
		p.potential = (float)rand() / RAND_MAX;
	}

    // Initialize rules - deterministic for stable forces
    for (Rule& r : rules) {
        r.attraction = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
		r.range = 1.0f + ((float)rand() / RAND_MAX) * 99.0f;
		r.power = ((float)rand() / RAND_MAX) * 5.5f - 0.5f;
    }
	for (float& m : mass) {
		m = ((float)rand() / RAND_MAX) * 499.9f + 0.1f;
	}
	for (float& e : energy_mul) {
		e = ((float)rand() / RAND_MAX) * 9.9f + 0.1f;
	}
	
    // Initialize CUDA managed memory buffers
    init_cuda_managed_buffers(num_particles, num_states, width, height);

    // Copy initial particles and rules into managed buffers
    memcpy(d_particles, particles.data(), sizeof(Particle) * num_particles);
    memcpy(d_rules, rules.data(), sizeof(Rule) * MAX_STATES * MAX_STATES);
	memcpy(d_mass, mass.data(), sizeof(float) * MAX_STATES);
	memcpy(d_energy_mul, energy_mul.data(), sizeof(float) * MAX_STATES);
	
    std::vector<uint32_t> framebuffer(width * height, 0);
	std::vector<uint32_t> accum(width * height, 0);
    bool quit = false;
	bool pause = false;
	bool voronoi2 = false;
	bool voronoi = false;
	bool flat = false;
	bool collide = false;
	bool trail = false;
	float threshold = 0.05;
	int particle_size = 1;
    SDL_Event e;

    while (!quit) {
		while (SDL_PollEvent(&e)) {
			switch (e.type) {
				case SDL_EVENT_QUIT:
					quit = true;
					break;
				case SDL_EVENT_KEY_DOWN:
					switch (e.key.key) {
						case SDLK_R:
							for (Rule& r : rules) {
								r.attraction = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
								r.range = 1.0f + ((float)rand() / RAND_MAX) * 99.0f;
								r.power = ((float)rand() / RAND_MAX) * 5.5f - 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 499.9f + 0.1f;
							}
							for (float& e : energy_mul) {
								e = ((float)rand() / RAND_MAX) * 9.9f + 0.1f;
							}
							memcpy(d_mass, mass.data(), sizeof(float) * MAX_STATES);
							memcpy(d_energy_mul, energy_mul.data(), sizeof(float) * MAX_STATES);
							memcpy(d_rules, rules.data(), sizeof(Rule) * MAX_STATES * MAX_STATES);
							break;
						case SDLK_B:
							for (Particle& p : particles) {
								p.x = ((float)rand() / RAND_MAX * width - width / 2) / zoom + width / 2 + offsetX;
								p.y = ((float)rand() / RAND_MAX * height - height / 2) / zoom + height / 2 + offsetY;
								p.vx = ((float)rand() / RAND_MAX - 0.5f);
								p.vy = ((float)rand() / RAND_MAX - 0.5f);
								p.state = rand() % num_states;
								p.energy = 1.0;
								p.potential = (float)rand() / RAND_MAX;
							}
							memcpy(d_particles, particles.data(), sizeof(Particle) * num_particles);
							break;
						case SDLK_P:
							zoom = 0.5f;
							offsetX = 0.0f;
							offsetY = 0.0f;
							break;
						case SDLK_MINUS:
						case SDLK_KP_MINUS:
							num_states = std::max(2, num_states - 1);
							for (Particle& p : particles) {
								p.state = std::min(p.state, num_states - 1);
							}
							memcpy(d_particles, particles.data(), sizeof(Particle) * num_particles);
							printf("Changed num_states to %d\n", num_states);
							break;
						case SDLK_EQUALS:
						case SDLK_KP_PLUS:
							num_states = std::min(MAX_STATES, num_states + 1);
							printf("Changed num_states to %d\n", num_states);
							break;
						case SDLK_UP:
							dt *= 1.1f;
							printf("Changed dt to %.7f\n", dt);
							break;
						case SDLK_DOWN:
							dt /= 1.1f;
							printf("Changed dt to %.7f\n", dt);
							break;
						case SDLK_RIGHT:
							max_velocity *= 1.1f;
							printf("Changed max_velocity to %.2f\n", max_velocity);
							break;
						case SDLK_LEFT:
							max_velocity /= 1.1f;
							printf("Changed max_velocity to %.2f\n", max_velocity);
							break;
						case SDLK_PERIOD:
							target_energy *= 1.1f;
							printf("Changed target_energy to %.5f\n", target_energy);
							break;
						case SDLK_COMMA:
							target_energy /= 1.1f;
							printf("Changed target_energy to %.5f\n", target_energy);
							break;
						case SDLK_2:
							frameskip++;
							printf("Changed frameskip to %d\n", frameskip);
							break;
						case SDLK_1:
							frameskip--;
							printf("Changed frameskip to %d\n", frameskip);
							break;
						case SDLK_4:
							potential_gain *= 1.1;
							printf("Changed potential_gain to %.5f\n", potential_gain);
							break;
						case SDLK_3:
							potential_gain /= 1.1;
							printf("Changed potential_gain to %.5f\n", potential_gain);
							break;
						case SDLK_SPACE:
							pause = !pause;
							break;
						case SDLK_G:
							particle_size--;
							printf("Changed particle_size to %d\n", particle_size);
							break;
						case SDLK_H:
							particle_size++;
							printf("Changed particle_size to %d\n", particle_size);
							break;
						case SDLK_5:
							voronoi = !voronoi;
							break;
						case SDLK_6:
							voronoi2 = !voronoi2;
							break;
						case SDLK_F:
							flat = !flat;
							break;
						case SDLK_7:
							threshold /= 1.1;
							printf("Changed threshold to %.3f\n", threshold);
							break;
						case SDLK_8:
							threshold *= 1.1;
							printf("Changed threshold to %.3f\n", threshold);
							break;
						case SDLK_C:
							collide = !collide;
							break;
						case SDLK_T:
							trail = !trail;
							break;
					}
					break;
				case SDL_EVENT_MOUSE_WHEEL:
					if (e.wheel.y > 0) zoom *= 1.1f;
					if (e.wheel.y < 0) zoom /= 1.1f;
					break;
				case SDL_EVENT_MOUSE_BUTTON_DOWN:
					if (e.button.button == SDL_BUTTON_LEFT) {
						dragging = true;
						lastMouseX = e.button.x;
						lastMouseY = e.button.y;
					} else if (e.button.button == SDL_BUTTON_RIGHT) {
						step_simulation(num_particles, num_states, dt, max_velocity, target_energy, potential_gain, collide, particle_size);
					}
					break;
				case SDL_EVENT_MOUSE_BUTTON_UP:
					if (e.button.button == SDL_BUTTON_LEFT) {
						dragging = false;
					}
					break;
				case SDL_EVENT_MOUSE_MOTION:
					if (dragging) {
						offsetX -= (e.motion.x - lastMouseX) / zoom;
						offsetY -= (e.motion.y - lastMouseY) / zoom;
						lastMouseX = e.motion.x;
						lastMouseY = e.motion.y;
					}
					break;
			}
		}
        // Run simulation step - d_particles updated in place in managed memory
		
		if (!pause) {
			for (int f = 0; f < frameskip; ++f) {
				step_simulation(num_particles, num_states, dt, max_velocity, target_energy, potential_gain, collide, particle_size);
			}
		}
		// Clear framebuffer to black
		memset(framebuffer.data(), 0, width * height * sizeof(uint32_t));
		if (!trail) {
			memset(accum.data(), 0, width * height * sizeof(uint32_t));
		}
		

		// Copy particle positions back to host vector for any CPU-side logic if needed
		memcpy(particles.data(), d_particles, sizeof(Particle) * num_particles);
		int num_visible = 0;
		std::vector<Particle> visible_particles;
		for (int i = 0; i < num_particles; ++i) {
			Particle& p = particles[i];
			int x = (int)((p.x - offsetX - width / 2.0f) * zoom + width / 2.0f);
			int y = (int)((p.y - offsetY - height / 2.0f) * zoom + height / 2.0f);
			if (x >= 0 && x < width && y >= 0 && y < height) {
				visible_particles.push_back(p);
				num_visible++;
			}
		}
		// Set each particle as a pixel
		if (voronoi) {
			dim3 threads(SMOOTH_BLOCK_SIZE, SMOOTH_BLOCK_SIZE);
			dim3 blocks((width + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE, (height + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE);
			std::copy(visible_particles.begin(), visible_particles.end(), d_visible);
			cudaMemset(d_input, 0, width * height * sizeof(uint32_t));
			render_voronoi<<<blocks, threads>>>(d_input, d_visible, num_visible, width, height, num_states, offsetX, offsetY, zoom, particle_size, d_mass, d_target_energy);
			cudaDeviceSynchronize();
		} else if (voronoi2) {
			dim3 threads(SMOOTH_BLOCK_SIZE, SMOOTH_BLOCK_SIZE);
			dim3 blocks((width + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE, (height + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE);
			std::copy(visible_particles.begin(), visible_particles.end(), d_visible);
			cudaMemset(d_input, 0, width * height * sizeof(uint32_t));
			render_voronoi2<<<blocks, threads>>>(d_input, d_visible, num_visible, width, height, num_states, offsetX, offsetY, zoom, particle_size, d_mass, d_target_energy, flat, threshold);
			cudaDeviceSynchronize();
		} else {
			for (int i = 0; i < num_visible; ++i) {
				Particle& p = visible_particles[i];
				int x = (int)((p.x - offsetX - width / 2.0f) * zoom + width / 2.0f);
				int y = (int)((p.y - offsetY - height / 2.0f) * zoom + height / 2.0f);
				if (x >= 0 && x < width && y >= 0 && y < height) {
					float h = (float)p.state / (float)num_states;
					float s = fminf(1.0f, p.energy / mass[p.state] / target_energy / 2) * 3 / 4 + 0.25;
					float v = fminf(1.0f, fmaxf(0.0f, 1.0f - p.potential / (p.energy * p.energy))) * 3 / 4 + 0.25;
					float c = v * s;
					float x_col = c * (1 - fabsf(fmodf(h * 6.0f, 2) - 1));
					float m = v - c;

					float r_, g_, b_;

					int sector = (int)(h * 6);
					switch (sector) {
						case 0: r_ = c; g_ = x_col; b_ = 0; break;
						case 1: r_ = x_col; g_ = c; b_ = 0; break;
						case 2: r_ = 0; g_ = c; b_ = x_col; break;
						case 3: r_ = 0; g_ = x_col; b_ = c; break;
						case 4: r_ = x_col; g_ = 0; b_ = c; break;
						case 5: default: r_ = c; g_ = 0; b_ = x_col; break;
					}
					float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
					if (x >= 0 && x < width && y >= 0 && y < height) {
						unpackRGBA(framebuffer[y * width + x], r, g, b, a);
						uint8_t r2 = (uint8_t)(fminf(255.0, fmaxf(0.0, r + (r_ + m)) * 255.0f));
						uint8_t g2 = (uint8_t)(fminf(255.0, fmaxf(0.0, g + (g_ + m)) * 255.0f));
						uint8_t b2 = (uint8_t)(fminf(255.0, fmaxf(0.0, b + (b_ + m)) * 255.0f));
						framebuffer[y * width + x] = (255 << 24) | (b2 << 16) | (g2 << 8) | r2;
					}
				}
			}
		}
		if (voronoi || voronoi2) {
			cudaMemcpy(framebuffer.data(), d_input, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		}
		int i = 0;
		if (trail) {
			dim3 threads(SMOOTH_BLOCK_SIZE, SMOOTH_BLOCK_SIZE);
			dim3 blocks((width + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE, (height + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE);
			cudaMemcpy(d_input, framebuffer.data(), width * height * sizeof(uint32_t), cudaMemcpyHostToDevice);
			add_trail<<<blocks, threads>>>(d_input, d_output, dt, width, height);
			cudaDeviceSynchronize();
			cudaMemcpy(accum.data(), d_output, width * height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			SDL_UpdateTexture(texture, nullptr, accum.data(), width * sizeof(uint32_t));
		} else {
			SDL_UpdateTexture(texture, nullptr, framebuffer.data(), width * sizeof(uint32_t));
		}
		
        SDL_RenderClear(renderer);
        SDL_RenderTexture(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        SDL_Delay(1000 / fps);
    }

    // Cleanup CUDA managed buffers
    free_cuda_managed_buffers();

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}