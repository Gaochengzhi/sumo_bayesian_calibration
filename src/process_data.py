import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import gaussian_kde
from scipy.stats import entropy


def filter_and_classify(pd_f, length_threshold=6):
    positive_velocity_groups = [group for _, group in pd_f.groupby("id") if all(group["xVelocity"] >= 0)]
    df = pd.concat(positive_velocity_groups)
    df["vehicleType"] = df["width"].map(lambda x: "bus" if x > length_threshold else "car")
    return df


def iqr_filter(data, column) -> pd.DataFrame:
    Q1 = data[column].quantile(0.05)
    Q3 = data[column].quantile(0.95)
    IQR = Q3 - Q1
    return data[(data[column] >= (Q1 - 1.5 * IQR)) & (data[column] <= (Q3 + 1.5 * IQR))]

def cal_kl_divergence(a_cache, b_cache, variables, vehicle_types):
    a_kde_data = a_cache["hist_kde_data"]
    b_kde_data = b_cache["hist_kde_data"]
    kl_list = []
    for vtype in vehicle_types:
        for variable in variables:
            cache_key = f"{vtype}_{variable}"
            if cache_key in a_kde_data and cache_key in b_kde_data:
                _,_,_, a_kde_x, a_kde_y = a_kde_data[
                    cache_key
                ]
                _,_,_, b_kde_x, b_kde_y = b_kde_data[
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

def get_all_kl_divergence(
    a_cache_path,
    b_cache_path,
    variables=["xAcceleration", "dhw", "xVelocity", "yVelocity"],
    vehicle_types=["car", "bus"],
):
    """
    input: a_path, b_path; output: kl_list
    returns a list of KL divergences for each variable and vehicle type
    """
    with open(a_cache_path, "rb") as f:
        a_cache = pickle.load(f)
    with open(b_cache_path, "rb") as f:
        b_cache = pickle.load(f)
    return cal_kl_divergence(a_cache, b_cache, variables, vehicle_types)



def save_distributions(
    df,
    output_dir="",
    variables=["xAcceleration", "dhw", "xVelocity"],
    vehicle_types=  ["car", "bus"]
):

    cache_file_path = os.path.join(output_dir, "_cache.pkl")
    hist_kde_data = {}
    stats_data = {}
    vehicle_count = {
        "car": df[df["vehicleType"] == "car"]["id"].nunique(),
        "bus": df[df["vehicleType"] == "bus"]["id"].nunique(),
    }
    stats_data["vehicle_count"] = vehicle_count

    for vtype in vehicle_types:
        vehicle_data = df[df["vehicleType"] == vtype]
        for variable in variables:
            filteredData = iqr_filter(vehicle_data, variable)
            if variable == "dhw":
                filteredData = filteredData[
                    filteredData[variable] > 0.2
                ]

            if not vehicle_data.empty:
                cache_key = f"{vtype}_{variable}"
                data = filteredData[variable].dropna()

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




def merge_data(base_dir, start_index=1, end_index=65):
    folders = [os.path.join(base_dir, f"DJI_{i:04d}") for i in range(start_index, end_index + 1)]
    
    def csv_path_exists(index, folder):
        path = os.path.join(folder, f"{index:02d}_tracks.csv")
        return path if os.path.isfile(path) else None
    
    csv_paths = filter(None, map(lambda t: csv_path_exists(t[0], t[1]), enumerate(folders, start=start_index)))
    df_list = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(df_list) if df_list else pd.DataFrame()


def compute_distribution(
    base_dir="../data",
    output_dir="../output",
    start_index=1,
    end_index=65,
    env="merge",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = merge_data(
        base_dir, start_index=start_index, end_index=end_index
    )
    filtered_data = filter_and_classify(df)
    save_distributions(filtered_data, output_dir, env=env)


if __name__ == "__main__":
    compute_distribution(start_index=1, end_index=8, env="merge")
    compute_distribution(start_index=15, end_index=16, env="stop")
    compute_distribution(start_index=18, end_index=23, env="right")
