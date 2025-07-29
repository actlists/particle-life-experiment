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

float zoom = 1.0f;
float offsetX = 0.0f, offsetY = 0.0f;
bool dragging = false;
int lastMouseX = 0, lastMouseY = 0;