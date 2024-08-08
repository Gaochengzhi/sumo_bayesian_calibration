import multiprocessing.process
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.core.problem import Problem
import time
import random
from task import SUMO_task, pbounds
from pymoo.algorithms.moo.nsga2 import NSGA2


class MooSUMOProblem(ElementwiseProblem):
    def __init__(self, param_bounds, env_name="merge", **kwargs):
        self.env_name = env_name
        self.param_bounds = param_bounds

        n_var = len(param_bounds)
        xl = [bounds[0] for bounds in param_bounds.values()]
        xu = [bounds[1] for bounds in param_bounds.values()]
        super().__init__(n_var=n_var, n_obj=8, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        params = {key: x[i] for i, key in enumerate(self.param_bounds.keys())}

        try:
            task = SUMO_task(params, env_name=self.env_name)
            res = task.run_task(sim_step=750 * 30, save=False, gui=False)
            if not res:
                out["F"] = [1] * self.n_obj
            else:
                out["F"] = res
        except Exception as e:
            out["F"] = [1] * self.n_obj


class SinSUMOProblem(ElementwiseProblem):
    def __init__(self, param_bounds, env_name="merge", **kwargs):
        self.env_name = env_name
        self.param_bounds = param_bounds
        n_var = len(param_bounds)
        xl = [bounds[0] for bounds in param_bounds.values()]
        xu = [bounds[1] for bounds in param_bounds.values()]
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        params = {key: x[i] for i, key in enumerate(self.param_bounds.keys())}

        try:
            task = SUMO_task(params, env_name=self.env_name)
            res = task.run_task(sim_step=750 * 30, save=False, gui=False)
            if not res:
                out["F"] = [1]
            else:
                out["F"] = [np.sum(res) / len(res)]
        except Exception as e:
            out["F"] = [1]


import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize


import pickle
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX

n_core = int(multiprocessing.cpu_count())


def run_optimization(problem, algorithm, algorithm_name):
    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 20),
        seed=1,
        save_history=True,
        verbose=True,
    )
    result_file = f"{algorithm_name}"
    with open(result_file, "wb") as f:
        pickle.dump(res, f)


def run_nsga2(problem):
    algorithm = NSGA2(pop_size=50)
    run_optimization(problem, algorithm, "../output/data_cache/nsga2_result.pkl")


def run_sms(problem):
    algorithm = SMSEMOA(pop_size=50)
    run_optimization(problem, algorithm, "../output/data_cache/sms_result.pkl")


def run_kgb(problem):
    algorithm = KGB(pop_size=50)
    run_optimization(problem, algorithm, "../output/data_cache/kgb_result.pkl")


def run_g3pcx(problem):
    algorithm = G3PCX(pop_size=50)
    run_optimization(problem, algorithm, "../output/data_cache/g3pcx_result.pkl")


from pymoo.algorithms.soo.nonconvex.pso import PSO


def run_pso(problem):
    algorithm = PSO(pop_size=50)
    run_optimization(problem, algorithm, "../output/data_cache/pso_result.pkl")


import multiprocessing
from pymoo.config import Config

Config.warnings["not_compiled"] = False

if __name__ == "__main__":
    pool = multiprocessing.Pool(n_core)
    runner = StarmapParallelization(pool.starmap)
    moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner)
    sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner)
    # run_g3pcx(moo_problem)
    run_kgb(moo_problem)
    # run_pso(sin_problem)
    # run_sms(moo_problem)
    # run_nsga2(moo_problem)

    # Create the problem with parallel evaluation
    pool.close()
