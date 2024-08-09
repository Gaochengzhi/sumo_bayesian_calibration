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
        super().__init__(n_var=n_var, n_obj=6, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        params = {key: x[i] for i, key in enumerate(self.param_bounds.keys())}

        try:
            task = SUMO_task(params, env=self.env_name)
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
            task = SUMO_task(params, env=self.env_name)
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
        termination=("n_gen", 30),
        seed=1,
        save_history=True,
        verbose=True,
    )
    result_file = f"../output/data_cache/{problem.env_name}_{algorithm_name}.pkl"
    with open(result_file, "wb") as f:
        pickle.dump(res, f)


from pymoo.algorithms.moo.age2 import AGEMOEA2


def run_age2(problem):
    algorithm = AGEMOEA2(pop_size=100)
    run_optimization(problem, algorithm, "age2")


from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions


# create the reference directions to be used for the optimization
def run_nsga3(problem):
    ref_dirs = get_reference_directions("energy", 6, 100, seed=1)
    algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
    run_optimization(problem, algorithm, "nsga3")


from pymoo.algorithms.soo.nonconvex.pso import PSO


def run_pso(problem):
    algorithm = PSO(pop_size=100)
    run_optimization(problem, algorithm, "pso.pkl")


import multiprocessing
from pymoo.config import Config

Config.warnings["not_compiled"] = False

if __name__ == "__main__":
    pool = multiprocessing.Pool(n_core)
    runner = StarmapParallelization(pool.starmap)
    moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name="right")
    sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name="right")
    # run_g3pcx(moo_problem)
    # run_kgb(moo_problem)
    # run_pso(sin_problem)
    # run_sms(moo_problem)
    # run_nsga3(moo_problem)
    # run_age2(moo_problem)
    moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name="stop")
    sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name="stop")
    run_nsga3(moo_problem)
    run_age2(moo_problem)

    # Create the problem with parallel evaluation
    pool.close()
