#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <thread>
#include "common.h"

#define BLOCK_SIZE 256

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

        float influence = expf(-dist * dist / (2.0f * rule.range * rule.range));
        float force = (mass[p_i.state] * mass[p_j.state]) / dist * ((rule.attraction + rule.power) * influence);
		
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
	float* target_energy,
	float* average_energy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    Particle& p = particles[i];
	if (p.potential > p.energy) {
		p.state = (p.state + 1) % num_states;
	}
	p.potential = (p.potential + p.energy) * dt + p.potential * (1 - dt * 2);
	p.energy += (mass[p.state] * max_velocity * max_velocity - p.energy) * dt;
}

static Particle* d_particles = nullptr;
static Rule* d_rules = nullptr;
static float* d_fx = nullptr;
static float* d_fy = nullptr;
static float* d_average_energy = nullptr;
static float* d_target_energy = nullptr;
static int* d_grid = nullptr;
static float* d_mass = nullptr;

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

    d_particles = nullptr;
    d_rules = nullptr;
    d_fx = nullptr;
    d_fy = nullptr;
    d_average_energy = nullptr;
    d_target_energy = nullptr;
    d_grid = nullptr;
	d_mass = nullptr;
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

    update_states<<<gridSize, blockSize>>>(d_particles, num_particles, num_states, dt, d_mass, max_velocity, d_target_energy, d_average_energy);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    int width = 1920;
    int height = 1080;
    int num_particles = 3072;
    int num_states = 3;
    int fps = 60;
    float dt = 0.05f;
    float max_velocity = 50.0f;
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
        p.vx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        p.vy = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        p.state = rand() % num_states;
        p.energy = (float)rand() / RAND_MAX;  // Start with positive energy
        p.potential = (float)rand() / RAND_MAX;
    }

    // Initialize rules - deterministic for stable forces
    for (Rule& r : rules) {
        r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 3.0f; // positive attraction for all pairs
		r.range = 80.0f - ((float)rand() / RAND_MAX) * 70.0f;
		r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
    }
	for (float& m : mass) {
		m = ((float)rand() / RAND_MAX) * 16.0f;
	}

    // Initialize CUDA managed memory buffers
    init_cuda_managed_buffers(num_particles, num_states, width, height);

    // Copy initial particles and rules into managed buffers
    memcpy(d_particles, particles.data(), sizeof(Particle) * num_particles);
    memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
	memcpy(d_mass, mass.data(), sizeof(float) * num_states);

    std::vector<uint32_t> framebuffer(width * height, 0);

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
								r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 5.0f - 0.5f; // positive attraction for all pairs
								r.range = 80.0f - ((float)rand() / RAND_MAX) * 70.0f;
								r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 16.0f;
							}
							memcpy(d_mass, mass.data(), sizeof(float) * num_states);
							memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
							break;
						case SDLK_B:
							for (Particle& p : particles) {
								p.x = static_cast<float>(rand() % width);
								p.y = static_cast<float>(rand() % height);
								p.vx = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
								p.vy = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
								p.state = rand() % num_states;
								p.energy = (float)rand() / RAND_MAX;
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
								r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 5.0f - 0.5f; // positive attraction for all pairs
								r.range = 80.0f - ((float)rand() / RAND_MAX) * 70.0f;
								r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 16.0f;
							}
							memcpy(d_mass, mass.data(), sizeof(float) * num_states);
							memcpy(d_rules, rules.data(), sizeof(Rule) * num_states * num_states);
							break;
						case SDLK_EQUALS:
						case SDLK_KP_PLUS:
							num_states++;
							rules = std::vector<Rule>(num_states * num_states);
							mass = std::vector<float>(num_states);
							for (Rule& r : rules) {
								r.attraction = ((float)rand() / RAND_MAX - 0.5f) * 5.0f - 0.5f; // positive attraction for all pairs
								r.range = 80.0f - ((float)rand() / RAND_MAX) * 70.0f;
								r.power = ((float)rand() / RAND_MAX) * 2.0f + 0.5f;
							}
							for (float& m : mass) {
								m = ((float)rand() / RAND_MAX) * 16.0f;
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
		
        // Copy framebuffer to SDL texture
		SDL_UpdateTexture(texture, nullptr, framebuffer.data(), width * sizeof(uint32_t));
		
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