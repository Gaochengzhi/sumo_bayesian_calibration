import time
import random
import uuid
import os
from collections import namedtuple
import shutil
from util import handle_exception, copy_files
import subprocess
from highway_env import run_one_sim
import pandas as pd
from process_data import (
    filter_and_classify,
    save_distributions,
    calculate_average_wasserstein_distance,
)


class SUMO_task:
    def __init__(self, param, env="merge"):
        ParamType = namedtuple("ParamType", param.keys())
        self.work_dir = None

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
        profile_template = f"""tau; normal({getattr(self.config, vehicle_type + '_tau_mean')},{getattr(self.config, vehicle_type + '_tau_std')});[1.5,3.8]
accel; {getattr(self.config, vehicle_type + '_acc')}
decel; {getattr(self.config, vehicle_type + '_dcc')}
maxSpeed; {getattr(self.config, vehicle_type + '_max_v')}
carFollowModel; EIDM
emergencyDecel; 5
lcSublane; {getattr(self.config, vehicle_type + '_lcSublane')}
lcPushy; {getattr(self.config, 'car_lcPushy')}
lcAssertive; 1
lcStrategic; 999
lcKeepRight; 0
lcOvertakeRight; 0
lcCooperative; {getattr(self.config, vehicle_type + '_lcCooperative')}
lcLookaheadLeft; 2
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

    def run_task(self, sim_step=754 * 30):
        run_one_sim(config_path=".", simulation_step=sim_step)

    def eval(self, data_name="merge"):
        pd_f = pd.read_csv("./record.csv")
        data = filter_and_classify(pd_f)
        save_distributions(
            data,
        )
        ## compare the file of ../output/f{data_name}_cache.pkl and ./_cache.pkl
        res = calculate_average_wasserstein_distance(
            "../../output/merge_cache.pkl", "_cache.pkl"
        )

        return res

    def close(self):
        if os.path.exists(f"../{self.task_id}"):
            shutil.rmtree(f"../{self.task_id}")

        os.chdir("../../src")
        return 0


def test():
    param = {
        "car_tau_mean": 1.6,
        "car_tau_std": 5,
        "bus_tau_mean": 2,
        "bus_tau_std": 10,
        "car_acc": 2,
        "car_dcc": 2,
        "bus_acc": 2,
        "bus_dcc": 2,
        "car_max_v": 12,
        "bus_max_v": 8,
        "car_lcSublane": 0.5,
        "bus_lcSublane": 0.5,
        "car_lcPushy": 0.5,
        "bus_lcPushy": 0.5,
        "car_lcCooperative": 0.5,
        "bus_lcCooperative": 0.5,
    }
    task = SUMO_task(param)
    task.run_task()
    res = task.eval()
    print(res)
    task.close()


if __name__ == "__main__":
    test()
