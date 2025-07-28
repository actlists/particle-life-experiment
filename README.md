# Particle Life with Energy Conservation and State Changes

## How to build


0. If you haven't, install Python 3.12\*. Then, install Cuda 12.9\*
1. Run `pip install numpy colorsys pybind11[global]` to install the Python dependencies needed for compilation and running.
2. Run `build.bat` to compile the code and generate the .pyd file needed to run the Python script.

\* <sub>Versions below or above probably work. I haven't tested it.</sub>

## Keyboard shortcuts

<kbd>R</kbd> - Randomize rules

<kbd>M</kbd> - Mutate rules

<kbd>S</kbd> - Save config

<kbd>L</kbd> - Load config

<kbd>↑</kbd> - Increase Delta Time

<kbd>↓</kbd> - Decrease Delta Time

<kbd>→</kbd> - Increase Maximum Velocity

<kbd>←</kbd> - Decrease Maximum Velocity

<kbd>Mouse Wheel ↑</kbd> - Increase Target Energy

<kbd>Mouse Wheel ↓</kbd> - Decrease Target Energy
