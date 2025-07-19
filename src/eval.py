import pickle
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from task import SUMO_task, pbounds
import json
from multi_object_optimization import MooSUMOProblem, SinSUMOProblem
from multiprocessing import Pool


def find_minimal_target_value(env, algo_list):
    min_F_vals = []
    min_X_vals = []

    for algo in algo_list:
        with open(f"../output/data_cache/{env}_{algo}.pkl", "rb") as f:
            res = pickle.load(f)
            history = res.history
            F_list = [entry.pop.get("F") for entry in history]
            F = np.vstack(F_list)
            mean_all_F = F.mean(axis=1)
            mini_F = mean_all_F.min(axis=0)
            min_F_vals.append(mini_F)
            min_index = mean_all_F.argmin(axis=0)
            X_list = [entry.pop.get("X") for entry in history]
            all_X = np.vstack(X_list)
            min_X = all_X[min_index]
            min_X_vals.append(min_X)

    min_index = np.argmin(min_F_vals)
    print(f'Minimal KL divwergence for environment "{env}": {min_F_vals[min_index]}')
    return min_X_vals[min_index]


def gen_eval_data(min_X, env_name):
    global pbounds
    param = {key: min_X[i] for i, key in enumerate(pbounds.keys())}
    task = SUMO_task(param, env=env_name)
    res = task.run_task(save=True, gui=False, sim_step=1200 * 30)
    print(res)


def gen_best_record(envs, algo_list):
    with Pool(processes=len(envs)) as pool:
        results = []
        for env in envs:
            min_X = find_minimal_target_value(env, algo_list)
            result = pool.apply_async(gen_eval_data, (min_X, env))
            results.append(result)
        # Ensure all tasks complete
        for result in results:
            result.wait()
        pool.close()
        pool.join()


if __name__ == "__main__":
    envs = ["merge", "stop", "right"]
    algo_list = ["nsga3", "age2", "pso"]
    gen_best_record(envs, algo_list)
