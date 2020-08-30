from phyre_utils import pic_to_action_vector, pic_hist_to_action
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools
from matplotlib import pyplot as plt
import os

def get_auccess(solver, tasks, solve_noise=False):
    eval_setup = 'ball_within_template'
    sim = phyre.initialize_simulator(tasks, 'ball')
    eva = phyre.Evaluator(tasks)

    # Get Actions from solver:
    all_actions = solver.get_actions(tasks)

    # Loop through actions
    for t_idx, task in enumerate(tasks):
        # Get 100 actions from solver
        if solve_noise:
            # expects one action for task
            task_actions = [all_actions[t_idx]]*100
        else:
            # expects 100 actions for task
            task_actions = all_actions[t_idx]
        # Loop through actions
        for j, action in enumerate(task_actions):
            # Simulate action
            res = sim.simulate_action(t_idx, action, need_featurized_objects=False)
            # Refine/Noise if invalid
            t = 0
            temp = 1
            base_action = action
            print(base_action, 'base action')       
            while res.status.is_invalid():
                t += 1
                action = base_action + (np.random.rand(3)-0.5)*0.01*temp
                res = sim.simulate_action(t_idx, action,  need_featurized_objects=False)
                temp *=1.01
                assert(t<500, "too many invalid tries")
            print(action, 'final action')
            # Log Attempt
            eva.maybe_log_attempt(t_idx, res.status)
    
    return eva.get_auccess()
    