
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.core.problem import Problem
import time
import random
import pymoo
from task import SUMO_task
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.default import DefaultMultiObjectiveTermination
import pickle

class SUMOProblem(ElementwiseProblem):
    def __init__(self, param_bounds, env_name="merge", **kwargs):
        self.env_name = env_name
        self.param_bounds = param_bounds
        
        # Number of variables is the length of parameter bounds
        n_var = len(param_bounds)
        # Lower and upper bounds for each of the variables
        xl = [bounds[0] for bounds in param_bounds.values()]
        xu = [bounds[1] for bounds in param_bounds.values()]

        super().__init__(n_var=n_var, n_obj=6, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        params = {key: x[i] for i, key in enumerate(self.param_bounds.keys())}
        
        try:
            task = SUMO_task(params, env_name=self.env_name)
            res = task.run_task(sim_step=750 * 30, save=False, gui=False)
            if not res:
                out["F"] = [1]*6
            else:
                out["F"] = res
        except Exception as e:
            out["F"] = [1]*6

with open('src/result1.pkl', 'rb') as f:
    res1 = pickle.load(f)
record  = res1
# get best result
group_50 = record.F
row_sums = np.sum(group_50, axis=1)
min_index = np.argmin(row_sums)
min_row = group_50[min_index]
res = record.history
print(res)
print(np.min(np.sum(group_50, axis=1))/6)
print(min_row)