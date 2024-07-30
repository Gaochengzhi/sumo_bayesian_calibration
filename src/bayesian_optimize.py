import multiprocessing
import tornado.ioloop
import tornado.web
from tornado.web import Application, RequestHandler
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import time
import random
import threading
import requests
import json
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from util import round_dic_data, handle_exception
from task import SUMO_task


def task_function(**params):
    # return random.uniform(1, 5)

    try:
        task = SUMO_task(params)
        task.run_task()
        res = task.eval()
        task.close()
    except Exception as e:
        handle_exception(e)
        res = 1
    return -res


class TaskHandler(RequestHandler):
    def initialize(
        self,
        task_queue,
        result_queue,
        optimizer,
        util,
        task_count,
        total_task_num,
        lock,
    ):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.optimizer = optimizer
        self.util = util
        self.task_count = task_count
        self.total_task_num = total_task_num
        self.lock = lock

    def post(self):
        result = tornado.escape.json_decode(self.request.body)
        params = result["params"]
        target = result["target"]

        with self.lock:
            self.optimizer.register(params=params, target=target)

            with self.task_count.get_lock():
                if self.task_count.value < self.total_task_num:
                    new_params = self.optimizer.suggest(self.util)
                    round_new_params = round_dic_data(new_params)
                    self.task_queue.put({"params": round_new_params})
                    self.task_count.value += 1
                else:
                    raise KeyboardInterrupt
        self.write({"status": "new task added"})


def make_app(
    task_queue, result_queue, optimizer, util, task_count, total_task_num, lock
):
    return Application(
        [
            (
                r"/task",
                TaskHandler,
                dict(
                    task_queue=task_queue,
                    result_queue=result_queue,
                    optimizer=optimizer,
                    util=util,
                    task_count=task_count,
                    total_task_num=total_task_num,
                    lock=lock,
                ),
            ),
        ]
    )


def execute_task(task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        params = task["params"]
        target = task_function(**params)

        print(target)
        result_queue.put({"params": params, "target": target})
        task_queue.task_done()


def result_handler(result_queue, task_handler_url):
    while True:
        result = result_queue.get()
        if result is None:
            break
        requests.post(
            task_handler_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(result),
        )
        result_queue.task_done()


# if __name__ == "__main__":
def bayesian_optimize(kp=5.576, xi=0.1, max_iteration_time=1000):
    pbounds = {
        "car_tau_mean": (1.6, 3),
        "car_tau_std": (0, 15),
        "bus_tau_mean": (2, 4),
        "bus_tau_std": (0, 15),
        "car_acc": (1, 3),
        "car_dcc": (1, 5),
        "bus_acc": (1, 2),
        "bus_dcc": (1, 4),
        "car_max_v": (8, 20),
        "bus_max_v": (8, 20),
        "car_lcSublane": (0, 1),
        "bus_lcSublane": (0, 1),
        "car_lcPushy": (0, 1),
        "bus_lcPushy": (0, 1),
        "car_lcCooperative": (0, 1),
        "bus_lcCooperative": (0, 1),
    }
    port_number = 8881
    lock = threading.Lock()

    date_time = str(time.strftime("%Y-%m-%d_%H:%M:%S"))
    logger = JSONLogger(path=f"../log/{date_time}_log.log")
    # cpu_count = int(multiprocessing.cpu_count() / 4)
    cpu_count = 12
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.JoinableQueue()
    task_count = multiprocessing.Value("i", 0)

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.set_gp_params(alpha=1e-2)
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
            target=execute_task, args=(task_queue, result_queue)
        )
        init_process.append(p)
        p.start()

    result_thread = threading.Thread(
        target=result_handler,
        args=(result_queue, f"http://localhost:{str(port_number)}/task"),
    )
    result_thread.daemon = True
    result_thread.start()

    app = make_app(
        task_queue,
        result_queue,
        optimizer,
        util,
        task_count,
        max_iteration_time,
        lock=lock,
    )
    app.listen(port_number)

    print(f"Server running on http://localhost:{str(port_number)}/")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("Shutting down.")
        for p in init_process:
            p.terminate()
        tornado.ioloop.IOLoop.current().stop()

    for p in init_process:
        p.join()
