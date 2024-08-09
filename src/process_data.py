import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import entropy

import json
import glob
from matplotlib.colors import LinearSegmentedColormap


def filter_and_classify(pd_f):
    grouped = pd_f.groupby("id")
    filtered_groups = []
    for _, group in grouped:
        if any(group["xVelocity"] < 0):
            continue
        filtered_groups.append(group)
    filtered_data = pd.concat(filtered_groups)
    filtered_data["vehicleType"] = filtered_data["width"].apply(
        lambda x: "bus" if x > 6 else "car"
    )
    return filtered_data


def iqr_filter(data, column):
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    return data[(data[column] >= (Q1 - 1.5 * IQR)) & (data[column] <= (Q3 + 1.5 * IQR))]


def kl_divergence(
    a_cache_path,
    b_cache_path,
    variables=["xAcceleration", "dhw", "xVelocity", "yVelocity"],
    vehicle_types=["car", "bus"],
):
    with open(a_cache_path, "rb") as f:
        a_cache = pickle.load(f)
    with open(b_cache_path, "rb") as f:
        b_cache = pickle.load(f)
    a_kde_data = a_cache["hist_kde_data"]
    b_kde_data = b_cache["hist_kde_data"]
    kl_list = []
    for vtype in vehicle_types:
        for variable in variables:
            cache_key = f"{vtype}_{variable}"
            if cache_key in a_kde_data and cache_key in b_kde_data:
                a_hist, a_bin_width, a_bin_centers, a_kde_x, a_kde_y = a_kde_data[
                    cache_key
                ]
                b_hist, b_bin_width, b_bin_centers, b_kde_x, b_kde_y = b_kde_data[
                    cache_key
                ]

                # Determine the common range over which to compare the two distributions
                min_range = max(a_kde_x.min(), b_kde_x.min())
                max_range = min(a_kde_x.max(), b_kde_x.max())

                if min_range < max_range:
                    common_x = np.linspace(min_range, max_range, 1000)

                    # Interpolate to get the KDE values on the common range
                    a_kde_common_y = np.interp(common_x, a_kde_x, a_kde_y)
                    b_kde_common_y = np.interp(common_x, b_kde_x, b_kde_y)

                    # Normalize the KDE values
                    a_kde_common_y /= np.sum(a_kde_common_y)
                    b_kde_common_y /= np.sum(b_kde_common_y)

                    # Calculate the KL divergence
                    kl = entropy(a_kde_common_y, b_kde_common_y)
                    kl_list.append(kl)
                else:
                    kl_list.append(float("inf"))

    return kl_list


def save_distributions(
    filtered_data,
    output_dir="",
    env="",
    variables=["xAcceleration", "dhw", "xVelocity"],
):

    cache_name = env + "_cache.pkl"
    cache_file_path = os.path.join(output_dir, cache_name)
    hist_kde_data = {}
    stats_data = {}
    vehicle_count = {
        "car": filtered_data[filtered_data["vehicleType"] == "car"]["id"].nunique(),
        "bus": filtered_data[filtered_data["vehicleType"] == "bus"]["id"].nunique(),
    }
    stats_data["vehicle_count"] = vehicle_count

    for vehicle_type in ["car", "bus"]:
        vehicle_data = filtered_data[filtered_data["vehicleType"] == vehicle_type]
        for variable in variables:
            filtered_vehicle_data = iqr_filter(vehicle_data, variable)
            if variable == "dhw":
                filtered_vehicle_data = filtered_vehicle_data[
                    filtered_vehicle_data[variable] > 0.2
                ]

            if not vehicle_data.empty:
                cache_key = f"{vehicle_type}_{variable}"
                data = filtered_vehicle_data[variable].dropna()

                if cache_key not in hist_kde_data:
                    kde = gaussian_kde(data)
                    kde_x = np.linspace(data.min(), data.max(), 1000)
                    kde_y = kde(kde_x)
                    hist_data, bins = np.histogram(data, bins=80, density=True)
                    bin_width = bins[1] - bins[0]
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    hist_kde_data[cache_key] = (
                        hist_data,
                        bin_width,
                        bin_centers,
                        kde_x,
                        kde_y,
                    )
                    stats_data[cache_key] = {
                        "min": data.min(),
                        "max": data.max(),
                        "mean": data.mean(),
                        "std": data.std(),
                    }

    with open(cache_file_path, "wb") as f:
        pickle.dump({"hist_kde_data": hist_kde_data, "stats_data": stats_data}, f)


def plot_distribution_from_cache(cache_file, output_dir):
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    combined_hist_kde_data = cache["hist_kde_data"]

    for cache_key, (
        hist_data,
        bin_width,
        bin_centers,
        kde_x,
        kde_y,
    ) in combined_hist_kde_data.items():
        vehicle_type, variable = cache_key.split("_")
        plot_distribution(
            kde_x,
            kde_y,
            hist_data,
            bin_centers,
            bin_width,
            vehicle_type,
            variable,
            output_dir,
        )


def load_stats_from_cache(cache_file):
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    stats_data = cache["stats_data"]
    return stats_data


def plot_distribution(
    kde_x, kde_y, hist_data, bin_centers, bin_width, vehicle_type, variable, output_dir
):
    plt.figure()
    plt.bar(
        bin_centers,
        height=hist_data,
        width=bin_width,
        align="center",
        edgecolor="k",
        # color="b",
        alpha=0.5,
    )
    plt.plot(kde_x, kde_y, label=f"KDE of {variable} for {vehicle_type}")
    plt.title(f"{variable} distribution for {vehicle_type}")
    plt.xlabel(variable)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    output_file = os.path.join(
        output_dir, f"{vehicle_type}_{variable}_distribution.png"
    )
    plt.savefig(output_file)
    plt.close()


def merge_data_from_folder(base_dir, start_index=1, end_index=65):
    folders = [
        os.path.join(base_dir, f"DJI_{i:04d}")
        for i in range(start_index, end_index + 1)
    ]
    df_list = [
        pd.read_csv(os.path.join(folder, f"{i:02d}_tracks.csv"))
        for i, folder in enumerate(folders, start=start_index)
        if os.path.isfile(os.path.join(folder, f"{i:02d}_tracks.csv"))
    ]
    return pd.concat(df_list) if df_list else pd.DataFrame()


def compute_distribution(
    base_dir="../data",
    output_dir="../output",
    start_index=18,
    end_index=23,
    env="merge",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene_pd = merge_data_from_folder(
        base_dir, start_index=start_index, end_index=end_index
    )
    filtered_data1 = filter_and_classify(scene_pd)
    save_distributions(filtered_data1, output_dir, env=env)


if __name__ == "__main__":
    # compute_distribution()
    compute_distribution(start_index=1, end_index=8, env="merge")
    compute_distribution(start_index=15, end_index=16, env="stop")
    compute_distribution(start_index=18, end_index=23, env="right")
    plot_distribution_from_cache(
        "../output/data_raw/merge/_cache.pkl", "../output/plot"
    )
    pass
