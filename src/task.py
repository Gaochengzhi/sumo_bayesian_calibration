"""
SUMO Task Management and Parameter Configuration

This module provides the core task management functionality for SUMO traffic simulation
calibration. It handles parameter configuration, workspace management, and simulation
execution with automatic cleanup.

Key Components:
- pbounds: Parameter boundaries for optimization algorithms
- SUMO_task: Main task class for managing individual simulation runs
- Utility functions for parameter extraction and evaluation
"""

import uuid
import os
from collections import namedtuple
import shutil
from util import handle_exception, copy_files, get_latest_file, json2pd
import subprocess
from highway_env import run_calibrate_sim
import pandas as pd
from process_data import (
    filter_and_classify,
    save_distributions,
    get_all_kl_divergence,
)


# Parameter bounds for optimization algorithms
# These bounds define the search space for 26 key traffic simulation parameters
pbounds = {
    # Car following model parameters
    "car_tau_mean": (0.5, 4),       # Car reaction time mean (seconds)
    "car_tau_std": (0, 10),         # Car reaction time std deviation
    "bus_tau_mean": (0.5, 4),       # Bus reaction time mean (seconds)
    "bus_tau_std": (0, 10),         # Bus reaction time std deviation
    
    # Acceleration and deceleration parameters
    "car_acc": (0.2, 4),            # Car maximum acceleration (m/s²)
    "car_dcc": (0.2, 4),            # Car maximum deceleration (m/s²)
    "bus_acc": (0.2, 4),            # Bus maximum acceleration (m/s²)
    "bus_dcc": (0.2, 4),            # Bus maximum deceleration (m/s²)
    
    # Driver imperfection parameters
    "car_sigma": (0, 1),            # Car driving imperfection (0=perfect, 1=imperfect)
    "car_lcSigma": (0, 1),          # Car lane-changing imperfection
    "bus_sigma": (0, 1),            # Bus driving imperfection
    "bus_lcSigma": (0, 1),          # Bus lane-changing imperfection
    
    # Speed distribution parameters
    "car_v_mean": (8, 26),          # Car desired speed mean (m/s)
    "car_v_std": (0, 20),           # Car desired speed std deviation
    "bus_v_mean": (8, 26),          # Bus desired speed mean (m/s)
    "bus_v_std": (0, 20),           # Bus desired speed std deviation
    
    # Lane-changing behavior parameters
    "car_lcSublane": (0, 1),        # Car sublane model parameter
    "bus_lcSublane": (0, 1),        # Bus sublane model parameter
    "car_lcPushy": (0, 1),          # Car pushy lane-changing behavior
    "bus_lcPushy": (0, 1),          # Bus pushy lane-changing behavior
    "car_lcSpeedGainRight": (0, 5), # Car speed gain threshold for right LC
    "bus_lcSpeedGainRight": (0, 5), # Bus speed gain threshold for right LC
    "car_lcAssertive": (1, 100),    # Car assertiveness in lane-changing
    "bus_lcAssertive": (1, 100),    # Bus assertiveness in lane-changing
    "car_lcCooperative": (0, 1),    # Car cooperative lane-changing
    "bus_lcCooperative": (0, 1),    # Bus cooperative lane-changing
    "car_lcLookaheadLeft": (2, 100), # Car lookahead distance for left LC
    "bus_lcLookaheadLeft": (2, 100), # Bus lookahead distance for left LC
}


class SUMO_task:
    """
    SUMO simulation task with automatic workspace management and evaluation.
    
    Args:
        param (dict): Simulation parameters
        env (str): Traffic scenario ('merge', 'stop', 'right')
    """
    
    def __init__(self, param, env="merge"):
        ParamType = namedtuple("ParamType", param.keys())
        self.work_dir = None
        self.env = env
        self.config = ParamType(**param)
        try:
            self.init_work_space(env)
        except Exception as e:
            handle_exception(e)
            self.close()

    def init_work_space(self, env):
        task_id = uuid.uuid4()
        self.task_id = task_id
        self.work_dir = f"../tmp/{task_id}"
        os.mkdir(self.work_dir)

        files_to_copy = [
            # "background.png",
            "stop.xml",
            "background.xml",
            "highway.net.xml",
            "highway.sumocfg",
            "autoGenTraffic.sh",
        ]
        copy_files(files_to_copy, f"../env/{env}", self.work_dir)
        self.create_vehicle_config(self.work_dir, "car")
        self.create_vehicle_config(self.work_dir, "bus")
        self.createVtypes()
        return task_id

    def create_vehicle_config(self, work_dir, vehicle_type):
        vtype = "passenger" if vehicle_type == "car" else vehicle_type
        profile_template = f"""tau; normal({getattr(self.config, vehicle_type + '_tau_mean')},{getattr(self.config, vehicle_type + '_tau_std')});[0.2,6]
accel; {getattr(self.config, vehicle_type + '_acc')}
decel; {getattr(self.config, vehicle_type + '_dcc')}
maxSpeed; normal({getattr(self.config, vehicle_type + '_v_mean')},{getattr(self.config, vehicle_type + '_v_std')});[6,30]
carFollowModel; EIDM
sigma; {getattr(self.config, vehicle_type + '_sigma')}
lcSigma; {getattr(self.config, vehicle_type + '_lcSigma')}
lcSublane; {getattr(self.config, vehicle_type + '_lcSublane')}
lcPushy; {getattr(self.config, vehicle_type + '_lcPushy')}
lcAssertive; {getattr(self.config, vehicle_type + '_lcAssertive')}
lcSpeedGainRight; {getattr(self.config, vehicle_type + '_lcSpeedGainRight')}
vClass; {vtype}
lcStrategic; 99
maxSpeedLat; 3.5
lcKeepRight; 0
lcOvertakeRight; 0
lcCooperative; {getattr(self.config, vehicle_type + '_lcCooperative')}
lcLookaheadLeft; {getattr(self.config, vehicle_type + '_lcLookaheadLeft')}
"""
        config_file_path = os.path.join(work_dir, f"{vehicle_type}.config.txt")
        with open(config_file_path, "w") as config_file:
            config_file.write(profile_template)
            config_file.flush()
            os.fsync(config_file.fileno())

    def createVtypes(self):
        os.chdir(self.work_dir)
        script_path = os.path.join(os.getcwd(), "autoGenTraffic.sh")
        if not os.access(script_path, os.X_OK):
            os.chmod(script_path, 0o755)
        try:
            result = subprocess.run(
                [script_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.stderr:
                print("Script Error Output:")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            handle_exception(e)
            self.close()

    def run_task(self, sim_step, save=False, gui=False):
        try:
            run_calibrate_sim(config_path=".", sim_step=sim_step, gui=gui)
            res = self.eval()
            if save:
                shutil.copytree(
                    ".", f"../../output/data_raw/{self.env}", dirs_exist_ok=True
                )
            return res
        except Exception as e:
            handle_exception(e)
        finally:
            self.close()
            os.chdir("../../src")

    def eval(self):
        compare_data_name = self.env
        pd_f = pd.read_csv("./record.csv")
        data = filter_and_classify(pd_f)
        save_distributions(
            data,
        )
        res = get_all_kl_divergence(
            f"../../output/data_cache/{compare_data_name}_cache.pkl",
            "_cache.pkl",
            variables=["xAcceleration", "dhw", "xVelocity"],
        )
        return res

    def close(self):
        if os.path.exists(f"../{self.task_id}"):
            shutil.rmtree(f"../{self.task_id}")
        return 0


def get_best_param(log_path=""):
    if not log_path:
        log_path = get_latest_file(folder="../log", suffix=".log")

    df = json2pd("../log/" + log_path + ".log")
    max_target_row = df.loc[df["target"].idxmax()]
    params_dic = {
        key: value for key, value in max_target_row.items() if key != "target"
    }
    return params_dic, -max_target_row["target"]


def eval_data(env):
    param, _ = get_best_param(env)
    task = SUMO_task(param, env=env)
    res = task.run_task(save=True, gui=False, sim_step=800 * 30)
    print("target:", res)


def manual_eval_tuning(env="merge"):
    pd_f = pd.read_csv(f"../output/data_raw/{env}/record.csv")
    data = filter_and_classify(pd_f)
    save_distributions(
        data,
        output_dir=f"../output/data_raw/{env}",
    )
    res = get_all_kl_divergence(
        f"../output/data_cache/{env}_cache.pkl",
        f"../output/data_raw/{env}/_cache.pkl",
        variables=["xAcceleration", "dhw", "xVelocity"],
    )
    print(res)
    print("Total KL divergence: " + str(res / len(res)))


from multiprocessing import Pool


def helper(env):
    run_calibrate_sim(config_path=f"../output/data_raw/{env}_origin", sim_step=800 * 30)
    manual_eval_tuning(env + "_origin")


if __name__ == "__main__":
    envs = ["merge", "right", "stop"]
    with Pool(processes=len(envs)) as pool:
        results = []
        for env in envs:
            r = pool.apply_async(helper, (env,))
            results.append(r)
        for r in results:
            r.wait()
        pool.close()
        pool.join()
