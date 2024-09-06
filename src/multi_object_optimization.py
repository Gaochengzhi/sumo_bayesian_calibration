from pymoo.core.problem import ElementwiseProblem
import numpy as np
from task import SUMO_task, pbounds
import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
import multiprocessing
import pickle

from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions


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


def run_age2(problem):
    algorithm = AGEMOEA2(pop_size=100)
    run_optimization(problem, algorithm, "age2")


def run_nsga3(problem):
    ref_dirs = get_reference_directions("energy", 6, 100, seed=1)
    algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
    run_optimization(problem, algorithm, "nsga3")


def run_pso(problem):
    algorithm = PSO(pop_size=100)
    run_optimization(problem, algorithm, "pso.pkl")


if __name__ == "__main__":

    n_core = int(multiprocessing.cpu_count())
    n_core = 101
    pool = multiprocessing.Pool(n_core)
    runner = StarmapParallelization(pool.starmap)

    # moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name="merge")
    # sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name="merge")
    # run_pso(sin_problem)
    # run_nsga3(moo_problem)
    # run_age2(moo_problem)
    moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name="right")
    sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name="right")
    run_pso(sin_problem)
    run_nsga3(moo_problem)
    run_age2(moo_problem)
    # moo_problem = MooSUMOProblem(pbounds, elementwise_runner=runner, env_name="stop")
    # sin_problem = SinSUMOProblem(pbounds, elementwise_runner=runner, env_name="stop")
    # run_pso(sin_problem)
    # run_nsga3(moo_problem)
    # run_age2(moo_problem)

    pool.close()
