import multiprocessing
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import time
import threading
import json
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from util import round_dic_data, handle_exception, get_latest_file, json2pd, params_to_tuple
from task import SUMO_task
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil


def task_function(**params):
    try:
        task = SUMO_task(params)
        id = task.task_id
        res = task.run_task(sim_step=104 * 30, save=False, gui=False)
        return res
    except Exception as e:
        handle_exception(e)
        return None


def execute_task(task_queue, result_queue, task_done_event):
    while not task_done_event.is_set():
        task = task_queue.get()
        if task is None:
            break
        params = task["params"]
        try:
            target = task_function(**params)
            if target is not None:
                result_queue.put({"params": params, "target": - target})
        except Exception as e:
            # handle_exception(e)
            print(e)
        finally:
            task_queue.task_done()


def result_handler(result_queue, optimizer, util, task_queue, task_count, total_task_num, task_done_event, lock, issued_params_set):
    while not task_done_event.is_set():
        result = result_queue.get()
        if result is None:
            break
        params = result["params"]
        target = result["target"]

        with lock:
            optimizer.register(params=params, target=target)
            with task_count.get_lock():
                while task_count.value < total_task_num:
                    new_params = optimizer.suggest(util)
                    round_new_params = round_dic_data(new_params)
                    # Check for duplicates
                    if params_to_tuple(round_new_params) not in issued_params_set:
                        task_queue.put({"params": round_new_params})
                        issued_params_set.add(
                            params_to_tuple(round_new_params))
                        task_count.value += 1
                        break
                    else:
                        print(f"Duplicate params detected: {
                              round_new_params}. Skipping new task.")
                else:
                    task_done_event.set()
        result_queue.task_done()


def bayesian_optimize(kp=5.576, xi=0.1, max_iteration_time=500):
    pbounds = {
        "car_tau_mean": (0.5, 3),
        "car_tau_std": (0, 15),
        "bus_tau_mean": (0.5, 3),
        "bus_tau_std": (0, 15),
        "car_acc": (1, 3),
        "car_dcc": (1, 5),
        "bus_acc": (1, 2),
        "bus_dcc": (1, 4),
        "car_max_v": (10, 33),
        "bus_max_v": (10, 23),
        "car_lcSublane": (0, 1),
        "bus_lcSublane": (0, 1),
        "car_lcPushy": (0, 1),
        "bus_lcPushy": (0, 1),
        "car_lcAssertive": (1, 100),
        "bus_lcAssertive": (1, 100),
        "car_lcCooperative": (0, 1),
        "bus_lcCooperative": (0, 1),
        "car_lcLookaheadLeft": (2, 100),
        "bus_lcLookaheadLeft": (2, 100),
    }
    lock = threading.Lock()
    issued_params_set = set()

    date_time = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
    logger = JSONLogger(path=f"../log/{date_time}_log.log")
    print(f"Init log file: {date_time}")
    cpu_count = int(multiprocessing.cpu_count() - 1)
    # cpu_count = 12
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.JoinableQueue()
    task_done_event = multiprocessing.Event()
    task_count = multiprocessing.Value("i", 0)

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.set_gp_params(alpha=1e-3)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    util = UtilityFunction(kind="ucb", kappa=kp, xi=xi)

    # init number of cpu_count tasks
    for _ in range(cpu_count):
        with lock:
            initial_params = optimizer.suggest(util)
        task_queue.put({"params": initial_params})

    init_process = []
    for _ in range(cpu_count):
        p = multiprocessing.Process(
            target=execute_task, args=(
                task_queue, result_queue, task_done_event)
        )
        init_process.append(p)
        p.start()

    result_thread = threading.Thread(
        target=result_handler,
        args=(result_queue, optimizer, util, task_queue, task_count,
              max_iteration_time, task_done_event, lock, issued_params_set),
    )
    result_thread.daemon = True
    result_thread.start()

    print(f"Starting Bayesian Optimization with {
          cpu_count} parallel processes.")

    try:
        task_queue.join()
        result_queue.join()
    except KeyboardInterrupt:
        print("Shutting down.")
        task_done_event.set()

    for p in init_process:
        task_queue.put(None)
        p.join()


def plot_iteration_score(log_path=""):
    if not log_path:
        log_path = get_latest_file(folder="../log", suffix=".log")

    df = json2pd(log_path)
    df["cummax_target"] = df["target"].cummax()
    plt.figure(figsize=(10, 6))
    plt.plot(-df["cummax_target"], label="Cumulative Max Target")
    # plt.plot(-df["target"], label="Current Target", marker="o")
    # plt.plot(-df["target"].expanding().mean(), label="Expanding Mean")
    plt.title("Iteration Scores")
    plt.xlabel("Iteration")
    plt.ylabel("Target Score")
    plt.grid(True)

    # Save the figure
    output_path = "../output/plot/iterative_convergence.png"
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    # bayesian_optimize()
    plot_iteration_score()
