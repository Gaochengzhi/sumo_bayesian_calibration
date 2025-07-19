import gym
import numpy as np
import os
import sys
from util import handle_exception
from sumolib import checkBinary
import traci
import csv


class Traffic_Env(gym.Env):

    def __init__(self, record_area="E3", config_path="."):
        self.record_area = record_area
        self.config_path = config_path
        self.record_path = None
        self.record_file = None
        self.csv_writer = None

    def step(self):
        traci.simulationStep()

    def start(
        self,
        gui=False,
        record=True,
    ):
        sumoBinary = checkBinary("sumo-gui") if gui else checkBinary("sumo")
        traci.start([sumoBinary, "-c", self.config_path + "/highway.sumocfg"])

        if record:
            self.record_path = self.config_path + "/record.csv"
            self.record_file = open(self.record_path, "w")
            self.writer = csv.writer(self.record_file)
            self.writer.writerow(
                [
                    "frame",
                    "id",
                    "width",
                    "xVelocity",
                    "yVelocity",
                    "xAcceleration",
                    "dhw",
                ]
            )

    def close(self):
        self.record_file.close()
        traci.close()

    def record(self, step):
        v_ids = traci.edge.getLastStepVehicleIDs(self.record_area)
        for vid in v_ids:
            xVelocity = round(traci.vehicle.getSpeed(vid), 3)
            yVelocity = round(traci.vehicle.getLateralSpeed(vid), 3)
            xAcceleration = round(traci.vehicle.getAcceleration(vid), 3)
            v_lenghth = round(traci.vehicle.getLength(vid), 3)
            _, dhw = traci.vehicle.getFollower(vid)
            dhw = round(dhw, 3)
            self.writer.writerow(
                [step, vid, v_lenghth, xVelocity, yVelocity, xAcceleration, dhw]
            )


def run_calibrate_sim(
    recording_area="E3",
    config_path="",
    sim_step=30 * (900 + 100),
    gui=False,
):
    env = Traffic_Env(record_area=recording_area, config_path=config_path)

    env.start(gui=gui, record=True)
    try:
        hot_time = 200 * 30
        for i in range(sim_step):
            if i > hot_time:
                env.record(i)
            env.step()
    except Exception as e:
        handle_exception(e)
    finally:
        traci.close()
        sys.stdout.flush()


if __name__ == "__main__":
    run_calibrate_sim(config_path="../output/data_raw/merge")
