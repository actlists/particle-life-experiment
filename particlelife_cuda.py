from tkinter import filedialog
import particlelife_cuda
import tkinter as tk
import numpy as np
import colorsys
import pygame
import math
import json

# Simulation parameters
WIDTH, HEIGHT = 1920, 1080
NUM_PARTICLES = 1000
NUM_STATES = 3
FPS = 60
DT = 0.05
MAX_VELOCITY = 20.0
TARGET_ENERGY = 1.0
MUTATION_STD = 0.1

# Create particle array (NumPy structured dtype matching C++ `Particle`)
particles_dtype = np.dtype([
    ('x', 'f4'), ('y', 'f4'),
    ('vx', 'f4'), ('vy', 'f4'),
    ('state', 'i4'), ('energy', 'f4'), ('potential', 'f4')
])
particles = np.zeros(NUM_PARTICLES, dtype=particles_dtype)

# Initialize with random positions, velocities, and states
particles['x'] = np.random.uniform(0, WIDTH, NUM_PARTICLES)
particles['y'] = np.random.uniform(0, HEIGHT, NUM_PARTICLES)
particles['vx'] = np.random.uniform(-1, 1, NUM_PARTICLES)
particles['vy'] = np.random.uniform(-1, 1, NUM_PARTICLES)
particles['state'] = np.random.randint(0, NUM_STATES, NUM_PARTICLES)
particles['energy'] = np.random.uniform(0, 1, NUM_PARTICLES)
particles['potential'] = np.random.uniform(0, 1, NUM_PARTICLES)

# Create rule matrix (NUM_STATES x NUM_STATES)
rules_dtype = np.dtype([
    ('attraction', 'f4'),
    ('radius', 'f4'),
    ('power', 'f4'),
])
rules = np.zeros((NUM_STATES, NUM_STATES), dtype=rules_dtype)

# Initialize with random rules
for i in range(NUM_STATES):
    for j in range(NUM_STATES):
        rules[i, j]['attraction'] = np.random.uniform(-2.0, 2.0)
        rules[i, j]['radius'] = np.random.uniform(5.0, 30.0)
        rules[i, j]['power'] = np.random.uniform(0.25, 2)

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont('Arial', 24)
pygame.display.set_caption("ParticleLife Viewer")
clock = pygame.time.Clock()
log_messages = []

def log(message):
    print(message)
    log_messages.append([0, message])
    if len(log_messages) > 10:
        log_messages.pop(0)

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=(("JSON files", "JSON *.json"))
    )

    if file_path:
        log(f"Selected file: {file_path.split("/\\")[-1]}")
        return file_path
    else:
        log("No file selected.")
        return

running = True
it = 0
while running:
    clock.tick(FPS)
    # Handle input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                for i in range(NUM_STATES):
                    for j in range(NUM_STATES):
                        rules[i, j]['attraction'] = np.random.uniform(-2.0, 2.0)
                        rules[i, j]['radius'] = np.random.uniform(5.0, 30.0)
                        rules[i, j]['power'] = np.random.uniform(0.25, 2)
                log(f"Randomized rules")
            elif event.key == pygame.K_m:
                for i in range(NUM_STATES):
                    for j in range(NUM_STATES):
                        rules[i, j]['attraction'] += np.random.uniform(-1.0, 1.0) * MUTATION_STD
                        rules[i, j]['radius'] = np.random.uniform(-1, 1) * MUTATION_STD
                        rules[i, j]['power'] = np.random.uniform(-1, 1) * MUTATION_STD
                log(f"Mutated rules")
            elif event.key == pygame.K_b:
                particles['x'] = np.random.uniform(0, WIDTH, NUM_PARTICLES)
                particles['y'] = np.random.uniform(0, HEIGHT, NUM_PARTICLES)
                particles['vx'] = np.random.uniform(-1, 1, NUM_PARTICLES)
                particles['vy'] = np.random.uniform(-1, 1, NUM_PARTICLES)
                particles['state'] = np.random.randint(0, NUM_STATES, NUM_PARTICLES)
                particles['energy'] = np.random.uniform(0, 1, NUM_PARTICLES)
                particles['potential'] = np.random.uniform(0, 1, NUM_PARTICLES)
                log(f"Randomized particles")
            elif event.key == pygame.K_UP and (DT + 0.01) <= 1:
                DT += 0.01
                log(f"Delta time changed to: {DT:.2f}")
            elif event.key == pygame.K_DOWN and (DT - 0.01) >= 0:
                DT -= 0.01
                log(f"Delta time changed to: {DT:.2f}")
            elif event.key == pygame.K_RIGHT and (MAX_VELOCITY + 2) <= 50:
                MAX_VELOCITY += 2
                log(f"Max velocity changed to: {MAX_VELOCITY:.0f}")
            elif event.key == pygame.K_LEFT and (MAX_VELOCITY - 2) >= 0:
                MAX_VELOCITY -= 2
                log(f"Max velocity changed to: {MAX_VELOCITY:.0f}")
            elif event.key == pygame.K_s:
                log("Saving configuration")
                config = {
                    "DT": DT,
                    "TARGET_ENERGY": TARGET_ENERGY,
                    "MAX_VELOCITY": MAX_VELOCITY,
                    "RULES": {
                        "ATTRACTION": rules['attraction'].tolist(),
                        "RADIUS": rules['radius'].tolist(),
                        "POWER": rules['power'].tolist()
                    }
                }
                filename = open_file_dialog()
                if filename is not None:
                    with open('particle_config.json', 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=4, ensure_ascii=False)
            elif event.key == pygame.K_l:
                log("Loading configuration")
                filename = open_file_dialog()
                if filename is not None:
                    with open(filename, 'r') as f:
                        config = json.load(f)
                        DT = config["DT"]
                        TARGET_ENERGY = config["TARGET_ENERGY"]
                        MAX_VELOCITY = config["MAX_VELOCITY"]
                        rules['attraction'] = config["RULES"]["ATTRACTION"]
                        rules['radius'] = config["RULES"]["RADIUS"]
                        rules['power'] = config["RULES"]["POWER"]
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0 and (TARGET_ENERGY + 0.05) <= 3:
                TARGET_ENERGY += 0.05
                log(f"Target energy changed to: {TARGET_ENERGY:.2f}")
            elif event.y < 0 and (TARGET_ENERGY - 0.05) >= 0:
                TARGET_ENERGY -= 0.05
                log(f"Target energy changed to: {TARGET_ENERGY:.2f}")
    # Update simulation using CUDA
    particlelife_cuda.step_simulation(particles, rules.ravel(), DT, WIDTH, HEIGHT, MAX_VELOCITY, TARGET_ENERGY)

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw particles
    for p in particles:
        color = tuple(map(lambda e: int(e * 255), colorsys.hsv_to_rgb(p['state'] / NUM_STATES, max(0, min(1, p['energy'] / 2)) / 4 * 3 + 0.25, max(0, min(1, p['potential'] / math.sqrt(p['vx'] ** 2 + p['vy'] ** 2))) / 4 * 3 + 0.25)))
        pygame.draw.circle(screen, color, (int(p['x']), int(p['y'])), 2)
    y_offset = 6
    for i, (timer, message) in enumerate(log_messages):
        text_surface = font.render(message, True, (255, 255, 255)) # White text
        screen.blit(text_surface, (6, y_offset))
        y_offset += text_surface.get_height() + 5
        log_messages[i][0] += 1
        if timer >= FPS * 5:
            del log_messages[i]
    it += 1
    pygame.display.flip()
pygame.quit()
