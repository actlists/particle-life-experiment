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

<kbd>Comma</kbd> - Increase target energy

<kbd>Period</kbd> - Decrease target energy

### Visuals
<kbd>A</kbd> - Toggle approximation mode

<kbd>Mouse Wheel ↑</kbd> - Zoom in

<kbd>Mouse Wheel ↓</kbd> - Zoom out

<kbd>Mouse Drag</kbd> - Move