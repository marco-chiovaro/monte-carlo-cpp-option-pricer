# monte-carlo-cpp-option-pricer
High-performance Monte Carlo engine for pricing European and Asian options. The high-speed simulation core is implemented in C++, while Python is leveraged for statistical processing, sensitivity analysis (Greeks), and visual reporting.

---

## Context & Objectives

Developed as a Final Project for the B.Sc. in Computer Engineering at Sapienza University of Rome, this project bridges software engineering and financial mathematics. 

In Quantitative Finance and Asset Liability Management (ALM), pricing path-dependent derivatives requires significant computational power. This repository demonstrates how a hybrid architecture—combining native C++ performance with Python's analytical ecosystem—can solve computationally expensive pricing problems efficiently.

---

## Core Capabilities

### Quantitative & Mathematical Features
* **Stochastic Path Generation:** Simulates asset trajectories using Geometric Brownian Motion (GBM) powered by high-performance pseudo-random number generators (Xoshiro256**).
* **Multi-Asset & Path-Dependent Pricing:** Supports European options and path-dependent derivatives (e.g., Asian options) with exact payoff logic.
* **Sensitivities & Risk Analytics:** Calculates option Greeks (Delta, Gamma, Vega) and constructs 3D Option Value Surfaces to analyze market risk exposure.
* **Statistical Output & Visualization:** Generates convergence plots, distribution histograms, and risk-profile surfaces via Python's data analysis stack.

### High-Performance Software Architecture
* **Object-Oriented Design (C++17):** Clean, modular architecture separating stochastic dynamics, payoff algorithms, and execution engines.
* **OpenMP Parallelization:** Multi-threaded execution across CPU cores, optimizing Monte Carlo path generation for large-scale simulations ($10^6+$ paths).
* **Zero-Overhead Python Bindings:** Native C++ core exposed to Python via `pybind11`, allowing high-speed computational tasks to run inside flexible Python pipelines.
* **Complexity & Performance Profiling:** Automated benchmarks measuring execution speedup, memory usage, and $O(N)$ computational scaling.

---

## Project Structure
```text
MonteCarlo-Pricer-Engine
├── include/              # Header Files
|   ├── MonteCarlo.hpp
|   ├── Option.hpp
|   ├── Payoff.hpp
|   ├── Process.hpp
|   └── Xoshiro.hpp
├── src/                  # C++ Core Source Files
│   ├── MonteCarlo.cpp
│   └── bindings.cpp      
├── python/               # Data Analysis & Plotting Scripts
│   ├── plot.py
│   ├── analysis_greeks.py
│   ├── test_greeks.py
|   ├── test_speed.py
|   ├── convergence.py
|   └── complexity_analysis.py
├── docs/                 # Performance plots & 3D Surfaces
└── README.md

---

# Clone the repository
git clone [https://github.com/marco-chiovaro/monte-carlo-cpp-option-pricer.git](https://github.com/YOUR_USERNAME/monte-carlo-cpp-option-pricer.git)
cd monte-carlo-cpp-option-pricer

# Compile C++ source into a Python extension module via pybind11
g++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
    -fopenmp src/bindings.cpp -o montecarlo_cpp$(python3-config --extension-suffix)

---

## Author

* **Marco Chiovaro**
* B.Sc. in Computer Engineering — Sapienza University of Rome
* M.Sc. Candidate in Finance and Insurance (Financial Risk and Data Analysis) 
