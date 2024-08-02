import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_distribution_from_cache(cache_file1, cache_file2, output_dir):
    with open(cache_file1, "rb") as f1, open(cache_file2, "rb") as f2:
        cache1 = pickle.load(f1)
        cache2 = pickle.load(f2)

    combined_hist_kde_data1 = cache1["hist_kde_data"]
    combined_hist_kde_data2 = cache2["hist_kde_data"]

    for cache_key in combined_hist_kde_data1.keys():
        if cache_key in combined_hist_kde_data2:
            data1 = combined_hist_kde_data1[cache_key]
            data2 = combined_hist_kde_data2[cache_key]
            vehicle_type, variable = cache_key.split("_")
            plot_distribution_comparison(
                data1, data2, vehicle_type, variable, output_dir
            )


def plot_distribution_comparison(data1, data2, vehicle_type, variable, output_dir):
    hist_data1, bin_width1, bin_centers1, kde_x1, kde_y1 = data1
    hist_data2, bin_width2, bin_centers2, kde_x2, kde_y2 = data2

    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.bar(
        bin_centers1,
        height=hist_data1,
        width=bin_width1,
        align="center",
        edgecolor="k",
        alpha=0.5,
        color="blue",
        label="Dataset 1 Histogram",
    )
    plt.bar(
        bin_centers2,
        height=hist_data2,
        width=bin_width2,
        align="center",
        edgecolor="k",
        alpha=0.5,
        color="red",
        label="Dataset 2 Histogram",
    )

    # Plot KDEs
    plt.plot(kde_x1, kde_y1, color="blue", linestyle="--", label="Dataset 1 KDE")
    plt.plot(kde_x2, kde_y2, color="red", linestyle="--", label="Dataset 2 KDE")

    plt.title(f"{variable} distribution comparison for {vehicle_type}")
    plt.xlabel(variable)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    output_file = os.path.join(
        output_dir, f"{vehicle_type}_{variable}_distribution_comparison.png"
    )
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    plot_distribution_from_cache(
        "../output/data_raw/merge/_cache.pkl",
        "../output/data_cache/merge_cache.pkl",
        "../output/plot",
    )
