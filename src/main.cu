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
#define SMOOTH_BLOCK_SIZE 32

__global__ void compute_forces(
    Particle* particles,
    Rule* rules,
    int num_particles,
    int num_states,
    float* force_x,
    float* force_y,
	float* mass
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

        float influence = expf(-dist * dist / (2.0f * rule.range * rule.range)) * (p_i.energy + p_j.energy) * rule.power;
        float force = (mass[p_i.state] * dist + mass[p_j.state] * dist) / (mass[p_i.state] + mass[p_j.state]) * rule.attraction * influence;
		
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
	p.energy = (p.energy - p.energy / *average_energy * *target_energy) * dt;
	p.vx += force_x[i] * dt * p.energy;
	p.vy += force_y[i] * dt * p.energy;
	float speed = sqrtf(p.vx * p.vx + p.vy * p.vy);
	if (speed > max_velocity * dt * 100) {
		float scale = max_velocity / speed;
		p.vx *= scale;
		p.vy *= scale;
	}
    p.x += p.vx * dt * p.energy;
    p.y += p.vy * dt * p.energy;
}

__global__ void update_states(
    Particle* particles,
    int num_particles,
    int num_states,
	float dt,
	float* mass,
	int max_velocity,
	float* average_energy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
	if (p.potential / p.energy > (mass[p.state] * max_velocity * max_velocity)) {
		p.state = (p.state + 1) % num_states;
		p.energy = mass[p.state] * max_velocity * max_velocity;
		p.potential = 0;
	}
	for (int j = 0; j < num_particles; ++j) {
		if (i == j) continue;
		Particle p2 = particles[j];
		atomicAdd(&p.potential, p2.potential * dt / (mass[p2.state] * max_velocity * max_velocity) / num_particles);
	}
	p.energy += (mass[p.state] * max_velocity * max_velocity - p.energy - p.potential) * dt;
}

__device__ __host__ inline void unpackRGBA(uint32_t packed, float& r, float& g, float& b, float& a) {
    r = float((packed >> 24) & 0xFF);
    g = float((packed >> 16) & 0xFF);
    b = float((packed >> 8)  & 0xFF);
    a = float((packed)       & 0xFF);
}

__device__ __host__ inline uint32_t packRGBA(float r, float g, float b, float a) {
    uint32_t R = min(max(int(r + 0.5f), 0), 255);
    uint32_t G = min(max(int(g + 0.5f), 0), 255);
    uint32_t B = min(max(int(b + 0.5f), 0), 255);
    uint32_t A = min(max(int(a + 0.5f), 0), 255);
    return (R << 24) | (G << 16) | (B << 8) | A;
}

__device__ float gaussianWeight(int dx, int dy, float sigma) {
    float dist2 = dx * dx + dy * dy;
    return expf(-dist2 / (2.0f * sigma * sigma));
}

__global__ void smoothFromDitheredRGBA(
    const uint32_t* input,
    float* accum_r, float* accum_g, float* accum_b, float* accum_a,
    int width, int height, int blur_radius, float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint32_t packed = input[idx];
    if (packed == 0) return;

    float r, g, b, a;
    unpackRGBA(packed, r, g, b, a);

    for (int dy = -blur_radius; dy <= blur_radius; ++dy) {
        for (int dx = -blur_radius; dx <= blur_radius; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                float w = gaussianWeight(dx, dy, sigma);
                atomicAdd(&accum_r[nidx], r * w);
                atomicAdd(&accum_g[nidx], g * w);
                atomicAdd(&accum_b[nidx], b * w);
                atomicAdd(&accum_a[nidx], a * w);
            }
        }
    }
}

__global__ void composeRGBAImage(
    const float* accum_r, const float* accum_g,
    const float* accum_b, const float* accum_a,
    uint32_t* output, int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float r = accum_r[idx];
    float g = accum_g[idx];
    float b = accum_b[idx];
    float a = accum_a[idx];

    output[idx] = packRGBA(r, g, b, a);
}

static Particle* d_particles = nullptr;
static Rule* d_rules = nullptr;
static float* d_fx = nullptr;
static float* d_fy = nullptr;
static float* d_average_energy = nullptr;
static float* d_target_energy = nullptr;
static int* d_grid = nullptr;
static float* d_mass = nullptr;
static float* d_accum_a = nullptr;
static float* d_accum_r = nullptr;
static float* d_accum_g = nullptr;
static float* d_accum_b = nullptr;
static uint32_t* d_input = nullptr;
static uint32_t* d_output = nullptr;
void init_cuda_managed_buffers(int num_particles, int num_states, int width, int height) {
    if (!d_particles)
        cudaMallocManaged(&d_particles, sizeof(Particle) * num_particles);
    if (!d_rules)
        cudaMallocManaged(&d_rules, sizeof(Rule) * num_states * num_states);
    if (!d_fx)
        cudaMallocManaged(&d_fx, sizeof(float) * num_particles);
    if (!d_fy)
        cudaMallocManaged(&d_fy, sizeof(float) * num_particles);
    if (!d_average_energy)
        cudaMallocManaged(&d_average_energy, sizeof(float));
    if (!d_target_energy)
        cudaMallocManaged(&d_target_energy, sizeof(float));
    if (!d_grid)
        cudaMallocManaged(&d_grid, sizeof(int) * width * height);
	if (!d_mass)
		cudaMallocManaged(&d_mass, sizeof(float) * num_states);
	if (!d_accum_a) {
		cudaMallocManaged(&d_accum_a, width * height * sizeof(float));
	}
	if (!d_accum_r) {
		cudaMallocManaged(&d_accum_r, width * height * sizeof(float));
	}
	if (!d_accum_g) {
		cudaMallocManaged(&d_accum_g, width * height * sizeof(float));
	}
	if (!d_accum_b) {
		cudaMallocManaged(&d_accum_b, width * height * sizeof(float));
	}
	if (!d_input) {
		cudaMallocManaged(&d_input, width * height * sizeof(uint32_t));
	}
	if (!d_output) {
		cudaMallocManaged(&d_output, width * height * sizeof(uint32_t));
	}
}

void free_cuda_managed_buffers() {
    if (d_particles) cudaFree(d_particles);
    if (d_rules) cudaFree(d_rules);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_average_energy) cudaFree(d_average_energy);
    if (d_target_energy) cudaFree(d_target_energy);
    if (d_grid) cudaFree(d_grid);
	if (d_mass) cudaFree(d_mass);
	if (d_accum_a) cudaFree(d_accum_a);
	if (d_accum_r) cudaFree(d_accum_r);
	if (d_accum_g) cudaFree(d_accum_g);
	if (d_accum_b) cudaFree(d_accum_b);
	if (d_input) cudaFree(d_input);
	if (d_output) cudaFree(d_output);

    d_particles = nullptr;
    d_rules = nullptr;
    d_fx = nullptr;
    d_fy = nullptr;
    d_average_energy = nullptr;
    d_target_energy = nullptr;
    d_grid = nullptr;
	d_mass = nullptr;
	d_accum_a = nullptr;
	d_accum_r = nullptr;
	d_accum_g = nullptr;
	d_accum_b = nullptr;
	d_input = nullptr;
	d_output = nullptr;
}

void step_simulation(int num_particles, int num_states, float dt, float max_velocity, float target_energy) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (num_particles + blockSize - 1) / blockSize;

    // Initialize average energy and target energy in managed memory
    *d_average_energy = 0.0f;
    *d_target_energy = target_energy;

    compute_forces<<<gridSize, blockSize>>>(d_particles, d_rules, num_particles, num_states, d_fx, d_fy, d_mass);
    cudaDeviceSynchronize();

    get_avg<<<gridSize, blockSize>>>(d_particles, num_particles, d_average_energy);
    cudaDeviceSynchronize();

    integrate<<<gridSize, blockSize>>>(d_particles, d_fx, d_fy, num_particles, dt, max_velocity, d_target_energy, d_average_energy);
    cudaDeviceSynchronize();

    update_states<<<gridSize, blockSize>>>(d_particles, num_particles, num_states, dt, d_mass, max_velocity, d_average_energy);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    int width = 1920;
    int height = 1080;
    int num_particles = 10000;
    int num_states = 3;
    int fps = 60;
    float dt = 0.05f;
    float max_velocity = 20.0f;
    float target_energy = 1.0f;
	
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
    std::vector<Rule> rules(num_states * num_states);
	std::vector<float> mass(num_states);

    // Initialize particles with random positions, velocities, states, and energy
	srand(time(0));
    for (Particle& p : particles) {
		p.x = static_cast<float>(rand() % width);
		p.y = static_cast<float>(rand() % height);
		p.vx = ((float)rand() / RAND_MAX - 0.5f) * max_velocity;
		p.vy = ((float)rand() / RAND_MAX - 0.5f) * max_velocity;
		p.state = rand() % num_states;
		p.energy = 1.0;
		p.potential = (float)rand() / RAND_MAX;
	}

    // Initialize rules - deterministic for stable forces
    for (Rule& r : rules) {
        r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
		r.range = 5.0f + ((float)rand() / RAND_MAX) * 40.0f;
		r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
    }
	for (float& m : mass) {
		m = ((float)rand() / RAND_MAX) * 25.0f + 0.5f;
	}

    // Initialize CUDA managed memory buffers
    init_cuda_managed_buffers(num_particles, num_states, width, height);

    // Copy initial particles and rules into managed buffers
    memcpy(d_particles, particles.data(), sizeof(Particle) * num_particles);
    memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
	memcpy(d_mass, mass.data(), sizeof(float) * num_states);

    std::vector<uint32_t> framebuffer(width * height, 0);
	bool smooth = false;
    bool quit = false;
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
								r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
								r.range = 5.0f + ((float)rand() / RAND_MAX) * 40.0f;
								r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 25.0f + 0.5f;
							}
							memcpy(d_mass, mass.data(), sizeof(float) * num_states);
							memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
							break;
						case SDLK_B:
							for (Particle& p : particles) {
								p.x = static_cast<float>(rand() % width);
								p.y = static_cast<float>(rand() % height);
								p.vx = ((float)rand() / RAND_MAX - 0.5f) * max_velocity;
								p.vy = ((float)rand() / RAND_MAX - 0.5f) * max_velocity;
								p.state = rand() % num_states;
								p.energy = 1.0;
								p.potential = (float)rand() / RAND_MAX;
							}
							memcpy(d_particles, particles.data(), sizeof(Particle) * num_particles);
							break;
						case SDLK_P:
							zoom = 1.0f;
							offsetX = 0.0f;
							offsetY = 0.0f;
							dt = 0.05f;
							break;
						case SDLK_MINUS:
						case SDLK_KP_MINUS:
							num_states = std::max(2, num_states - 1);
							rules = std::vector<Rule>(num_states * num_states);
							mass = std::vector<float>(num_states);
							for (Rule& r : rules) {
								r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
								r.range = 5.0f + ((float)rand() / RAND_MAX) * 40.0f;
								r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 25.0f + 0.5f;
							}
							memcpy(d_mass, mass.data(), sizeof(float) * num_states);
							memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
							break;
						case SDLK_EQUALS:
						case SDLK_KP_PLUS:
							num_states++;
							rules = std::vector<Rule>(num_states * num_states);
							mass = std::vector<float>(num_states);
							for (Particle& p : particles) {
								p.state = min(p.state, num_states);
							}
							for (Rule& r : rules) {
								r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 3.0f;
								r.range = 80.0f - ((float)rand() / RAND_MAX) * 70.0f;
								r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 25.0f + 0.5f;
							}
							memcpy(d_mass, mass.data(), sizeof(float) * num_states);
							memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
							break;
						case SDLK_UP:
							dt *= 1.1f;
							printf("Changed dt to %.4f\n", dt);
							break;
						case SDLK_DOWN:
							dt /= 1.1f;
							printf("Changed dt to %.4f\n", dt);
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
							printf("Changed target_energy to %.2f\n", target_energy);
							break;
						case SDLK_COMMA:
							target_energy /= 1.1f;
							printf("Changed target_energy to %.2f\n", target_energy);
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
						smooth = !smooth;
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
        step_simulation(num_particles, num_states, dt, max_velocity, target_energy);

        // Copy particle positions back to host vector for any CPU-side logic if needed
        memcpy(particles.data(), d_particles, sizeof(Particle) * num_particles);

		// Clear framebuffer to black
		memset(framebuffer.data(), 0, width * height * sizeof(uint32_t));
		
		// Set each particle as a pixel
		for (int i = 0; i < num_particles; ++i) {
			Particle& p = particles[i];
			int x = (int)((p.x - offsetX - width / 2.0f) * zoom + width / 2.0f);
			int y = (int)((p.y - offsetY - height / 2.0f) * zoom + height / 2.0f);
			if (x >= 0 && x < width && y >= 0 && y < height) {
				float h = (float)p.state / (float)num_states;
				float s = 1.0f;
				float v = fminf(1.0f, p.energy);
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

				uint8_t r = (uint8_t)((r_ + m) * 255);
				uint8_t g = (uint8_t)((g_ + m) * 255);
				uint8_t b = (uint8_t)((b_ + m) * 255);
				framebuffer[y * width + x] = (255 << 24) | (b << 16) | (g << 8) | r;
			}
		}
		
		if (smooth) {
			cudaMemset(d_output, 0, width * height * sizeof(uint32_t));
			cudaMemcpy(d_input, framebuffer.data(), width * height * sizeof(uint32_t), cudaMemcpyHostToDevice);
			// Clear accumulators
			cudaMemset(d_accum_r, 0, sizeof(float) * width * height);
			cudaMemset(d_accum_g, 0, sizeof(float) * width * height);
			cudaMemset(d_accum_b, 0, sizeof(float) * width * height);
			cudaMemset(d_accum_a, 0, sizeof(float) * width * height);
			dim3 block(SMOOTH_BLOCK_SIZE, SMOOTH_BLOCK_SIZE);
			dim3 grid((width + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE, (height + SMOOTH_BLOCK_SIZE - 1) / SMOOTH_BLOCK_SIZE);
			// Smooth
			smoothFromDitheredRGBA<<<grid, block>>>(
				d_input, d_accum_r, d_accum_g, d_accum_b, d_accum_a,
				width, height, 16, 4.0
			);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("CUDA error after kernel 1: %s\n", cudaGetErrorString(err));
			}
			
			cudaDeviceSynchronize();

			// Final image
			composeRGBAImage<<<grid, block>>>(
				d_accum_r, d_accum_g, d_accum_b, d_accum_a, d_output, width, height
			);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("CUDA error after kernel 2: %s\n", cudaGetErrorString(err));
			}
			cudaDeviceSynchronize();
			SDL_UpdateTexture(texture, nullptr, d_output, width * sizeof(uint32_t));
		} else {
			// Copy framebuffer to SDL texture
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