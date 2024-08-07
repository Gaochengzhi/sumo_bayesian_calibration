#  2024.07.25 Code for calibrate the scene for AD4CHE dataset in SUMO simulation
# It involve the following steps:
# 1. Prepare the data in data folder like DJI_0001 to DJI_0065

from process_data import compute_distribution, load_stats_from_cache
from single_optimize import bayesian_optimize
from muti_object_optimization import run_nsga2, run_sms, SinSUMOProblem, pbounds


def main(base_dir="../data", output_dir="../output"):
    # 2. Generate statistic data cache in output for different scenes for comparison and inital gusss

    # compute_distribution(base_dir, output_dir)

    # 3. Optimize the scene for the best result
    bayesian_optimize(max_iteration_time=1000)



if __name__ == "__main__":
    run_sms()
    run_nsga2()
    main()
