import multiprocessing
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import time
import threading
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from util import (
    round_dic_data,
    handle_exception,
    params_to_tuple,
)
from task import SUMO_task, pbounds
import numpy as np


def task_function(env, **params):
    try:
        task = SUMO_task(params, env=env)
        res = task.run_task(sim_step=750 * 30, save=False, gui=False)
        return res
    except Exception as e:
        handle_exception(e)
        return None


def execute_task(task_queue, result_queue, task_done_event, env):
    while not task_done_event.is_set():
        task = task_queue.get()
        if task is None:
            break
        params = task["params"]
        try:
            target = task_function(**params, env=env)
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
                        issued_params_set.add(
                            params_to_tuple(round_new_params))
                        task_count.value += 1
                        break
                    else:
                        print(
                            f"Duplicate params detected: {round_new_params}. Skipping new task."
                        )
                else:
                    task_done_event.set()
        result_queue.task_done()


def bayesian_optimize(
    kp=4,
    xi=0.01,
    max_iteration=500,
    pbounds=pbounds,
    env="merge",
    log_name=None,
    cpu_count=int(multiprocessing.cpu_count()) - 4,
):
    if not log_name:
        log_name = env
    lock = threading.Lock()
    issued_params_set = set()
    date_time = str(time.strftime("%Y-%m-%d_%H:%M"))
    logger = JSONLogger(path=f"../log/{log_name}_{date_time}.log")
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

    for _ in range(cpu_count):
        with lock:
            initial_params = optimizer.suggest(util)
        task_queue.put({"params": initial_params})

    init_process = []
    for _ in range(cpu_count):
        p = multiprocessing.Process(
            target=execute_task, args=(
                task_queue, result_queue, task_done_event, env)
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
            max_iteration,
            task_done_event,
            lock,
            issued_params_set,
        ),
    )
    result_thread.daemon = True
    result_thread.start()

    print(
        f"Starting Bayesian Optimization with {cpu_count} parallel processes.")

    try:
        while task_count.value < max_iteration and not task_done_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        task_done_event.set()

    result_thread.join()
    print("All result handling finished.")
    for p in init_process:
        if p.is_alive():
            p.terminate()
    return


if __name__ == "__main__":
    # bayesian_optimize(max_iteration=3000, env="merge", log_name="merge")
    bayesian_optimize(max_iteration=3000, env="right", log_name="right")
    # bayesian_optimize(max_iteration=3000, env="stop", log_name="stop")
