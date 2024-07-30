#  2024.07.25 Code for calibrate the scene for AD4CHE dataset in SUMO simulation
# It involve the following steps:
# 1. Prepare the data in data folder like DJI_0001 to DJI_0065

from process_data import compute_distribution, load_stats_from_cache
from bayesian_optimize import bayesian_optimize


def main(base_dir="../data", output_dir="../output"):
    # 2. Generate statistic data cache in output for different scenes for comparison and inital gusss

    # compute_distribution(base_dir, output_dir)

    # 3. Optimize the scene for the best result
    bayesian_optimize(max_iteration_time=2000)


if __name__ == "__main__":
    main()
