@echo off
mkdir build
cd build
cmake ..\src
cmake --build .
move %~dp0build\debug\ParticleLifeExperiment.exe %~dp0
cd ..