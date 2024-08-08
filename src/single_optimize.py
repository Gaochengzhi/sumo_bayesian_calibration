import multiprocessing
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import time
import threading
import json
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from util import (
    round_dic_data,
    handle_exception,
    get_latest_file,
    json2pd,
    params_to_tuple,
)
from task import SUMO_task, pbounds
import os
import shutil
import numpy as np


def task_function(**params):
    try:
        task = SUMO_task(params)
        res = task.run_task(sim_step=750 * 30, save=False, gui=False)
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
                res = -np.sum(target) / len(target)
                result_queue.put({"params": params, "target": res})
        except Exception as e:
            handle_exception(e)
        finally:
            task_queue.task_done()


def result_handler(
    result_queue,
    optimizer,
    util,
    task_queue,
    task_count,
    total_task_num,
    task_done_event,
    lock,
    issued_params_set,
):
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
                        issued_params_set.add(params_to_tuple(round_new_params))
                        task_count.value += 1
                        break
                    else:
                        print(
                            f"Duplicate params detected: {round_new_params}. Skipping new task."
                        )
                else:
                    task_done_event.set()
        result_queue.task_done()


def bayesian_optimize(kp=4, xi=0.01, max_iteration_time=500, pbounds=pbounds):
    lock = threading.Lock()
    issued_params_set = set()

    date_time = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
    logger = JSONLogger(path=f"../log/{date_time}_log.log")
    print(f"Init log file: {date_time}")
    cpu_count = int(multiprocessing.cpu_count())
    # cpu_count = 2
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
            target=execute_task, args=(task_queue, result_queue, task_done_event)
        )
        init_process.append(p)
        p.start()

    result_thread = threading.Thread(
        target=result_handler,
        args=(
            result_queue,
            optimizer,
            util,
            task_queue,
            task_count,
            max_iteration_time,
            task_done_event,
            lock,
            issued_params_set,
        ),
    )
    result_thread.daemon = True
    result_thread.start()

    print(f"Starting Bayesian Optimization with {cpu_count} parallel processes.")

    try:
        while task_count.value < max_iteration_time and not task_done_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("============== Shutting down ===========")
        task_done_event.set()

    result_thread.join()
    print("All result handling finished.")
    for p in init_process:
        if p.is_alive():
            p.terminate()
    return


from multi_object_optimization import SinSUMOProblem, MooSUMOProblem, run_pso, run_kgb


from pymoo.core.problem import StarmapParallelization

if __name__ == "__main__":
    pool = multiprocessing.Pool(100)
    runner = StarmapParallelization(pool.starmap)
    problem1 = SinSUMOProblem(pbounds, elementwise_runner=runner)
    problem2 = MooSUMOProblem(pbounds, elementwise_runner=runner)
    # run_pso(problem1)
    # run_kgb(problem2)
    bayesian_optimize(max_iteration_time=1000)
    pool.close()
