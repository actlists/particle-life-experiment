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

float zoom = 0.5f;
float offsetX = 0.0f, offsetY = 0.0f;
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;
int width = 1280;
int height = 720;
int num_particles = 8192;
int num_states = 3;
int fps = 60;
int frameskip = 1;
float dt = 0.01f;
float max_velocity = 1000.0f;
float target_energy = 0.05f;
float potential_gain = 0.1f;