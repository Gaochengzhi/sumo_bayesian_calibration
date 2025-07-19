"""
SUMO Traffic Simulation Calibration Pipeline

This module implements a complete calibration pipeline for SUMO traffic simulation
using the AD4CHE dataset. It combines multiple optimization algorithms including
Bayesian optimization, NSGA-III, AGE-MOEA2, and PSO to calibrate traffic parameters
for three different scenarios: merge, stop, and right turn.

The pipeline involves the following steps:
1. Data preprocessing and distribution computation
2. Bayesian optimization for parameter tuning
3. Multi-objective optimization using evolutionary algorithms

Author: Gaochengzhi <Gaochengzhi1999@gmail.com>
Date: 2024.07.25
"""
from process_data import compute_distribution
from bayesian_optimize import bayesian_optimize
from multi_object_optimization import (
    MooSUMOProblem,
    SinSUMOProblem,
    run_age2,
    run_nsga3,
    run_pso,
)
from task import pbounds
import multiprocessing
from pymoo.core.problem import StarmapParallelization


def main(base_dir="../data", output_dir="../output"):
    """
    Main calibration pipeline for SUMO traffic simulation.
    
    Args:
        base_dir (str): Path to AD4CHE dataset directory
        output_dir (str): Path to output directory for results
    """
    # Step 1: Data preprocessing and distribution computation
    # Process AD4CHE dataset frames for different traffic scenarios
    distribution_tasks = [(1, 8, "merge"), (15, 16, "stop"), (18, 23, "right")]
    for start_idx, end_idx, scenario in distribution_tasks:
        compute_distribution(start_index=start_idx, end_index=end_idx, env=scenario)
    
    # Step 2: Initialize parallel processing for optimization
    # Reserve 4 cores for system stability
    n_core = int(multiprocessing.cpu_count() - 4)
    pool = multiprocessing.Pool(n_core)
    runner = StarmapParallelization(pool.starmap)
    
    # Step 3: Run optimization algorithms for each scenario
    for scenario in ["merge", "right", "stop"]:
        # Step 3.1: Bayesian optimization for parameter exploration
        bayesian_optimize(max_iteration=3000, env=scenario, cpu_count=n_core)
        
        # Step 3.2: Multi-objective optimization
        # Create problem instances for multi-objective and single-objective optimization
        moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name=scenario)
        sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name=scenario)
        
        # Run different optimization algorithms
        run_pso(sin_problem)      # Particle Swarm Optimization
        run_nsga3(moo_problem)    # Non-dominated Sorting Genetic Algorithm III
        run_age2(moo_problem)     # Age-based Multi-objective Evolutionary Algorithm II
    
    # Clean up resources
    pool.close()


if __name__ == "__main__":
    main()
