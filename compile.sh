#!/bin/bash
clear 
echo "Start compilation C++..."

g++ -O3 -Wall -shared -std=c++11 -march=native -ffast-math -fPIC \
    $(python3 -m pybind11 --includes) \
    -I ./include \
    src/MonteCarlo.cpp \
    src/bindings.cpp \
    -o python/montecarlo_cpp$(python3-config --extension-suffix) \
    -fopenmp

echo "Compilation completed! .so file generated."
