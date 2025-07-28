# Particle Life with Energy Conservation and State Changes

## How to build

0. If you haven't, install Python 3.12, CUDA 12.9, and CMake.\*
1. Run `pip install numpy pygame colorsys pybind11[global]` to install the Python dependencies needed for compilation and running.
2. Run `build.bat` to compile the code and generate the .pyd file needed to run the Python script.

\* <sub>Versions below or above probably work. I haven't tested it.</sub>

## Keyboard shortcuts

### Rules
<kbd>R</kbd> - Randomize rules

<kbd>M</kbd> - Mutate rules

### Save/Load
<kbd>S</kbd> - Save config

<kbd>L</kbd> - Load config

### Parameters
<kbd>↑</kbd> - Increase delta time

<kbd>↓</kbd> - Decrease delta time

<kbd>→</kbd> - Increase maximum velocity
						
<kbd>←</kbd> - Decrease maximum velocity

<kbd>Mouse Wheel ↑</kbd> - Increase target energy

<kbd>Mouse Wheel ↓</kbd> - Decrease target energy

### Visuals
<kbd>A</kbd> - Toggle approximation mode

<kbd>Comma</kbd> - Zoom in

<kbd>Period</kbd> - Zoom out

<kbd>Keypad 8</kbd> - Move up

<kbd>Keypad 2</kbd> - Move down

<kbd>Keypad 4</kbd> - Move left

<kbd>Keypad 6</kbd> - Move right