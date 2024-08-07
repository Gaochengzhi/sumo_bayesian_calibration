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
                out["F"] = [1]*8
            else:
                out["F"] = res
        except Exception as e:
            out["F"] = [1]*8

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
                out["F"] = [np.sum(res)/len(res)]
        except Exception as e:
            out["F"] = [1]


import multiprocessing
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

n_cpu = 100







import pickle
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.kgb import KGB

def run_optimization(problem, algorithm, result_file):

    
    
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 20),
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    with open(result_file, 'wb') as f:
        pickle.dump(res, f)
    

def run_nsga2(problem):
    algorithm = NSGA2(pop_size=50)
    run_optimization(problem,algorithm, 'result1.pkl')

def run_sms(problem):
    algorithm = SMSEMOA(pop_size=50)
    run_optimization(problem,algorithm, 'result2.pkl')

def run_kgb(problem):
    algorithm = KGB(pop_size=50)
    run_optimization(problem,algorithm, 'result3.pkl')


from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX


def run_g3pcx(problem):
    algorithm = G3PCX(pop_size=50)
    run_optimization(problem,algorithm, 'result4.pkl')

########## single
from pymoo.algorithms.soo.nonconvex.pso import PSO
def run_pso(problem):
    algorithm = PSO(pop_size=50)
    run_optimization(problem,algorithm, 'result5.pkl')
from pymoo.problems.single import Rastrigin


if __name__ == '__main__':
    pool = multiprocessing.Pool(n_cpu)
    runner = StarmapParallelization(pool.starmap)
    problem = MooSUMOProblem(pbounds, elementwise_runner=runner)
    run_nsga2(problem)
    run_sms(problem)
    run_kgb(problem)
    run_g3pcx(problem)
    pool.close()