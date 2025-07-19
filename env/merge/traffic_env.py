import gym
import numpy as np
import os
import sys
import math
import xml.dom.minidom

try:
    sys.path.append(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools")
    )  # tutorial in tests
    sys.path.append(
        os.path.join(
            os.environ.get(
                "SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")
            ),
            "tools",
        )
    )  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')"
    )

import traci
import csv
import argparse


class Traffic_Env(gym.Env):

    def __init__(self, record_area="E3", config_path="."):
        self.maxDistance = 200.0
        self.maxSpeed = 15.0
        self.max_angle = 360.0
        self.AutoCarID = "Auto"
        self.reset_times = 0
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
        record=False,
    ):
        sumoBinary = checkBinary("sumo-gui") if gui else checkBinary("sumo")
        traci.start([sumoBinary, "-c", self.config_path + "/highway.sumocfg"])

        if record:
            self.record_path = self.config_path + "/record.csv"
            self.record_file = open(self.record_path, "w")
            self.writer = csv.writer(self.record_file)
            self.writer.writerow(
                ["frame", "id", "xVelocity", "yVelocity", "xAcceleration", "dhw"]
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
            _, dhw = traci.vehicle.getFollower(vid)
            dhw = round(dhw, 3)
            self.writer.writerow([step, vid, xVelocity, yVelocity, xAcceleration, dhw])


def test():
    parser = argparse.ArgumentParser(description="Simulate a highway configuration")

    # 添加参数
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.basename("."),
    )
    parser.add_argument(
        "--recording_area",
        type=str,
        default="E3",
        help="vehicle that being recorded, default 'E3'",
    )
    parser.add_argument(
        "--simulation_time",
        type=int,
        default=30 * (279 + 100),
        help="simulation time, default 30 fps, 2790 s and 100 s for hot load",
    )

    args = parser.parse_args()
    env = Traffic_Env(record_area=args.recording_area, config_path=args.config_path)
    env.start(gui=False, record=True)
    for i in range(args.simulation_time):
        if i > 100 * 30:
            env.record(i)
        env.step()
    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    test()
