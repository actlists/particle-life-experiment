@echo off
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..\src
cmake --build . --config Release
move Release\particlelife_cuda.pyd ..\
cd ..