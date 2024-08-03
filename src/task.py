import time
import random
import uuid
import os
from collections import namedtuple
import shutil
from util import handle_exception, copy_files, get_latest_file, json2pd
import subprocess
from highway_env import run_one_sim
import pandas as pd
from process_data import (
    filter_and_classify,
    save_distributions,
    calculate_average_wasserstein_distance,
)


class SUMO_task:
    def __init__(self, param, env_name="merge"):
        ParamType = namedtuple("ParamType", param.keys())
        self.work_dir = None
        self.env_name = env_name

        self.config = ParamType(**param)
        try:
            self.init_work_space(env_name)

        except Exception as e:
            handle_exception(e)
            self.close()

    def init_work_space(self, env):
        task_id = uuid.uuid4()
        self.task_id = task_id
        self.work_dir = f"../tmp/{task_id}"
        os.mkdir(self.work_dir)

        files_to_copy = [
            "background.png",
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
        profile_template = f"""tau; normal({getattr(self.config, vehicle_type + '_tau_mean')},{getattr(self.config, vehicle_type + '_tau_std')});[0.1,4]
accel; {getattr(self.config, vehicle_type + '_acc')}
decel; {getattr(self.config, vehicle_type + '_dcc')}
maxSpeed; {getattr(self.config, vehicle_type + '_max_v')}
carFollowModel; EIDM
emergencyDecel; 5
lcSublane; {getattr(self.config, vehicle_type + '_lcSublane')}
lcPushy; {getattr(self.config, vehicle_type + '_lcPushy')}
lcAssertive; {getattr(self.config, vehicle_type + '_lcAssertive')}
vClass; {vtype}
lcStrategic; 999
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

    def run_task(self, sim_step=754 * 30, save=False, gui=False):
        try:
            run_one_sim(config_path=".", simulation_step=sim_step, gui=gui)
            res = self.eval()
            if save:
                shutil.copytree(
                    ".", f"../../output/data_raw/{self.env_name}", dirs_exist_ok=True
                )
            return res
        except Exception as e:
            handle_exception(e)
        finally:
            self.close()
            os.chdir("../../src")

    def eval(self, compare_data_name="merge"):
        pd_f = pd.read_csv("./record.csv")
        data = filter_and_classify(pd_f)
        save_distributions(
            data,
        )
        res = calculate_average_wasserstein_distance(
            f"../../output/data_cache/{compare_data_name}_cache.pkl", "_cache.pkl"
        )

        return res

    def close(self):
        if os.path.exists(f"../{self.task_id}"):
            shutil.rmtree(f"../{self.task_id}")
        return 0


def get_best_param(log_path=""):
    if not log_path:
        log_path = get_latest_file(folder="../log", suffix=".log")

    df = json2pd(log_path)
    max_target_row = df.loc[df["target"].idxmax()]
    params = {key: value for key, value in max_target_row.items()
              if key != "target"}
    return params


def gen_eval_data():
    param = get_best_param()
    task = SUMO_task(param)
    res = task.run_task(save=True, gui=False)
    print(res)


if __name__ == "__main__":
    gen_eval_data()
