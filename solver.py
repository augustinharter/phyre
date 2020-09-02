from phyre_utils import pic_to_action_vector, pic_hist_to_action
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

def get_auccess(solver, tasks, solve_noise=False, save_tries=False):
    if save_tries:
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)

    eval_setup = 'ball_within_template'
    sim = phyre.initialize_simulator(tasks, 'ball')
    eva = phyre.Evaluator(tasks)

    # Get Actions from solver:
    all_actions = solver.get_actions(tasks)
    #print(list(zip(tasks, all_actions)))
    #return 0

    # Loop through actions
    for t_idx, task in enumerate(tasks):
        # Get 100 actions from solver
        if solve_noise:
            # expects one action for task
            task_actions = [all_actions[t_idx]]
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
                #assert(t>500, "too many invalid tries")
            print(action, 'valid action')
            # Log Attempt
            eva.maybe_log_attempt(t_idx, res.status)

            if solve_noise:
                base_action = action
                temp = 1
                flag = False
                error = False
                t = 0
                while eva.attempts_per_task_index[t_idx]<100:
                    if not res.status.is_solved():
                        action = base_action + (np.random.rand(3)-0.5)*np.array([0.3,0.05,0.05])*temp
                        res = sim.simulate_action(t_idx, action,  need_featurized_objects=False)
                        temp *=1.01
                        eva.maybe_log_attempt(t_idx, res.status)
                        t += 1
                        if t > 1000:
                            if not flag:
                                print(f"WARNING can't find valid action for {task}")
                                flag = True
                                error = True
                            eva.maybe_log_attempt(t_idx, phyre.SimulationStatus.NOT_SOLVED)      
                    else:
                        if not flag:
                            print(f"{task} solved after", eva.attempts_per_task_index[t_idx])
                        flag = True
                        eva.maybe_log_attempt(t_idx, res.status)
            if not flag:
                print(f"{task} solved after", eva.attempts_per_task_index[t_idx]+1)
            if save_tries and not j and not error:
                for k in range(len(res.images)):
                    scene = res.images[k]
                    img = Image.fromarray(phyre.observations_to_uint8_rgb(scene))
                    #draw = ImageDraw.Draw(img)
                    #draw.text((0, 0), f"{k-i_step} {tuple(delta*256)}", (15, 15, 15), font=font)

                    os.makedirs(f'result/solver/pyramid/', exist_ok=True)
                    img.save(f'result/solver/pyramid/{task}_{k}.png')
    
    return eva.get_auccess()

if __name__ == "__main__":
    from flownet import *
    solver = FlownetSolver("pyramid-CONS", "pyramid")

    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids+dev_ids+test_ids
    print("ready to get auccess")
    filter_list = ['00000', '00001', '00002', '00003', '00004','00005', 
                    '00006','00007','00008','00009','00010','00011',
                    '00012','00013','00014','00015','00016','00017',
                    '00018','00019','00020','00021','00022','00023', '00024']
    """
    auccess_per_task = dict()
    for filter_task in filter_list:
        selected_tasks = [t for t in all_tasks if t[:5]==filter_task]
        auccess = get_auccess(solver, selected_tasks, solve_noise=True, save_tries=True)
        auccess_per_task[filter_task] = auccess
    print(auccess_per_task)
    """
    auccess = get_auccess(solver, all_tasks, solve_noise=True, save_tries=True)
    print(auccess)