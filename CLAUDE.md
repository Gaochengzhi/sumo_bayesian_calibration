# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a SUMO (Simulation of Urban MObility) traffic simulation calibration project that uses Bayesian optimization and multi-objective optimization techniques to calibrate traffic parameters against real-world data from the AD4CHE dataset.

## Key Commands

### Running the Main Pipeline
```bash
cd src
python main.py
```

### Running Individual Components
```bash
# Run Bayesian optimization only
python bayesian_optimize.py

# Run multi-objective optimization
python multi_object_optimization.py

# Process and analyze data
python process_data.py

# Evaluate specific scenarios
python task.py

# Generate visualization plots
python render_plot.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Note: Requires SUMO traffic simulator to be installed separately
```

## Code Architecture

### Core Components

1. **main.py** - Main pipeline orchestrator that runs the complete calibration workflow:
   - Data distribution computation for different scenarios (merge, stop, right turn)
   - Bayesian optimization for parameter tuning
   - Multi-objective optimization using NSGA-III, AGE-MOEA2, and PSO algorithms

2. **task.py** - Core task management and SUMO simulation wrapper:
   - `SUMO_task` class handles individual simulation runs
   - `pbounds` dictionary defines parameter bounds for optimization
   - Creates temporary workspaces for each simulation run
   - Generates vehicle configuration files based on parameters

3. **highway_env.py** - SUMO simulation environment:
   - `Traffic_Env` class wraps SUMO simulation
   - Handles simulation execution and data recording
   - Records vehicle dynamics (velocity, acceleration, headway distance)

4. **bayesian_optimize.py** - Bayesian optimization implementation:
   - Uses `bayes_opt` library for parameter optimization
   - Parallel execution support for multiple CPU cores
   - Minimizes KL divergence between simulated and real data

5. **multi_object_optimization.py** - Multi-objective optimization:
   - `MooSUMOProblem` for multi-objective optimization (6 objectives)
   - `SinSUMOProblem` for single-objective optimization
   - Supports NSGA-III, AGE-MOEA2, and PSO algorithms

6. **process_data.py** - Data processing and analysis:
   - Filters and classifies vehicle data
   - Calculates KL divergence between distributions
   - Generates statistical distributions for comparison

### Data Structure

- `data/` - Raw video data (AD4CHE dataset)
- `env/` - SUMO simulation configurations for different scenarios:
  - `merge/` - Highway merge scenario
  - `stop/` - Stop sign scenario  
  - `right/` - Right turn scenario
- `output/` - Generated results and cache files
- `log/` - Optimization logs and results

### Key Parameters

The system optimizes 26 traffic parameters including:
- Vehicle following model parameters (tau, acceleration, deceleration)
- Speed distributions (mean, std)
- Lane changing behavior (sigma, assertiveness, cooperation)
- Vehicle-specific parameters for cars and buses

### Scenarios

Three main traffic scenarios are calibrated:
1. **merge** - Highway merge scenario (frames 1-8)
2. **stop** - Stop sign scenario (frames 15-16)
3. **right** - Right turn scenario (frames 18-23)

## Development Notes

- The project uses temporary directories for each simulation run to avoid conflicts
- SUMO simulations run for 750*30 steps (about 750 seconds of simulation time)
- Results are cached as pickle files for faster subsequent analysis
- The system supports both GUI and headless SUMO execution
- Multiprocessing is used extensively for parallel optimization

## Translation and Localization Notes

- 注意除了中文文档其余都写英文