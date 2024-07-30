import os
import pickle
import numpy as np
from scipy.stats import wasserstein_distance


def calculate_average_wasserstein_distance(a_cache_path, b_cache_path):
    with open(a_cache_path, "rb") as f:
        a_cache = pickle.load(f)

    with open(b_cache_path, "rb") as f:
        b_cache = pickle.load(f)

    a_kde_data = a_cache["hist_kde_data"]
    b_kde_data = b_cache["hist_kde_data"]

    variables = ["xVelocity", "yVelocity", "xAcceleration", "dhw"]

    avg_distances = []

    for vehicle_type in ["car", "bus"]:
        for variable in variables:
            cache_key = f"{vehicle_type}_{variable}"
            if cache_key in a_kde_data and cache_key in b_kde_data:
                a_kde_x, a_kde_y = a_kde_data[cache_key][3], a_kde_data[cache_key][4]
                b_kde_x, b_kde_y = b_kde_data[cache_key][3], b_kde_data[cache_key][4]
                common_x = np.linspace(
                    min(a_kde_x[0], b_kde_x[0]), max(a_kde_x[-1], b_kde_x[-1]), 1000
                )
                a_interpolated_y = np.interp(common_x, a_kde_x, a_kde_y)
                b_interpolated_y = np.interp(common_x, b_kde_x, b_kde_y)

                distance = wasserstein_distance(a_interpolated_y, b_interpolated_y)
                avg_distances.append(distance)

    return np.mean(avg_distances)


a_cache_path = "../output/merge_cache.pkl"
b_cache_path = "../output/merge_cache.pkl"
average_distance = calculate_average_wasserstein_distance(a_cache_path, b_cache_path)
print("average Wasserstein distance:", average_distance)
