#  2024.07.25 Code for calibrate the scene for AD4CHE dataset in SUMO simulation

# It involve the following steps:


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
    # 1. Prepare the data in data folder like DJI_0001 to DJI_0065
    # /data/DJI_0001 ~ /data/DJI_0065
    # 2. Generate statistic data cache in output for different scenes for comparison and inital gusss
    distribution_tasks = [
        (1, 8, "merge"),
        (15, 16, "stop"),
        (18, 23, "right")
    ]
    for i, j, env in distribution_tasks:
        compute_distribution(start_index=i, end_index=j, env=env)

    # 3. Optimize the scene for the best result
    n_core = int(multiprocessing.cpu_count() - 4)
    pool = multiprocessing.Pool(n_core)
    runner = StarmapParallelization(pool.starmap)
    for env in ["merge", "right", "stop"]:
    #   3.1 bayesian optimization
        bayesian_optimize(max_iteration=3000, env=env, cpu_count=n_core)
    #   3.2 multi-object optimization and pso
        moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name=env)
        sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name=env)
        run_pso(sin_problem)
        run_nsga3(moo_problem)
        run_age2(moo_problem)
    pool.close()


if __name__ == "__main__":
    main()
