from phyre_utils import pic_to_action_vector, pic_hist_to_action, vis_batch, make_mono_dataset, grow_action_vector, action_delta_generator
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import sys

def get_auccess(solver, tasks, solve_noise=False, save_tries=False, brute=False):
    if save_tries:
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)

    eval_setup = 'ball_within_template'
    sim = phyre.initialize_simulator(tasks, 'ball')
    init_scenes = T.tensor([[cv2.resize((scene==channel).astype(float), (32,32)) for channel in range(2,7)] for scene in sim.initial_scenes]).float().flip(-2)
    eva = phyre.Evaluator(tasks)

    # Get Actions from solver:
    if brute:
        all_actions = solver.get_actions(tasks, init_scenes, brute =True)
    else:
        all_actions = solver.get_actions(tasks, init_scenes)
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
            # Setting up visualization array
            vis_wid = 64
            vis_stack = T.zeros(6,10,vis_wid,vis_wid,3)
            vis_count = 1

            # Simulate action
            res = sim.simulate_action(t_idx, action, need_featurized_objects=False)  

            # Refining if invalid Action
            t = 0
            temp = 1
            base_action = action.copy()
            print(base_action, 'base action')     
            # Checking for valid action
            while res.status.is_invalid():
                t += 1
                action = base_action + (np.random.rand(3)-0.5)*0.05*temp
                print(action, f"potential action for task {task}")
                res = sim.simulate_action(t_idx, action,  need_featurized_objects=False)
                temp *= 1.01 if temp <5 else 1
                #assert(t>500, "too many invalid tries")
            print(action, 'valid action')

            # Log first Attempt
            eva.maybe_log_attempt(t_idx, res.status)
            # Visualizing first attempt
            if save_tries:
                for i in range(min(len(res.images), 10)):
                    vis_stack[0,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))

            # Collecting 100 Actions if solve noise
            warning_flag = False
            if solve_noise:
                base_action = action
                temp = 1
                error = False
                t = 0
                delta_generator = action_delta_generator()

                # Looping while less then 100 attempts
                while eva.attempts_per_task_index[t_idx]<100:
                    # Searching for new action while not solved
                    if not res.status.is_solved():
                        """ OLD APPROACH
                        action = base_action + (np.random.rand(3)-0.5)*np.array([0.3,0.05,0.05])*temp
                        temp *= 1.01 if temp <5 else 1
                        """
                        if t<1000:
                            action = base_action + delta_generator.__next__()
                            res = sim.simulate_action(t_idx, action,  need_featurized_objects=False)
                            eva.maybe_log_attempt(t_idx, res.status)
                            t += 1
                        else:
                            if not warning_flag:
                                print(f"WARNING can't find valid action for {task}")
                                warning_flag = True
                                error = True
                            eva.maybe_log_attempt(t_idx, phyre.SimulationStatus.NOT_SOLVED)

                    # if solved -> repeating action
                    else:
                        if not warning_flag:
                            print(f"{task} solved after", eva.attempts_per_task_index[t_idx])

                            # Visualization
                            if save_tries and not error:
                                for i in range(min(len(res.images), 10)):
                                    vis_stack[5,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                        warning_flag = True
                        eva.maybe_log_attempt(t_idx, res.status)
                    
                    # Visualization
                    if save_tries and not error and not res.status.is_invalid() and t and vis_count<5:
                        for i in range(min(len(res.images), 10)):
                            vis_stack[vis_count,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                        vis_count +=1

                if not warning_flag and not res.status.is_solved() and eva.attempts_per_task_index[t_idx]==100:
                    print(f"{task} not solved")
                vis_batch(vis_stack, f'result/solver/pyramid', f"{task}_attempts")
            # Not Solve Noise Case
            else:
                # Visualization
                if save_tries and not res.status.is_invalid() and vis_count<5:
                    for i in range(min(len(res.images), 10)):
                        vis_stack[vis_count,i] = T.tensor(cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid,vis_wid)))
                    vis_count +=1
                if res.status.is_solved():
                    print(f"{task} solved after", eva.attempts_per_task_index[t_idx])
                    vis_batch(vis_stack, f'result/solver/pyramid', f"{task}_attempts")
                    while eva.attempts_per_task_index[t_idx]<100:
                        eva.maybe_log_attempt(t_idx, res.status)
                    break
    
    return eva.get_auccess()

if __name__ == "__main__":
    from flownet import *

    model_path = sys.argv[sys.argv.index("--path")+1] if "--path" in sys.argv else "default"
    print("Model path:", model_path)
    epochs = int(sys.argv[sys.argv.index("--epochs")+1]) if "--epochs" in sys.argv else 10
    nper = int(sys.argv[sys.argv.index("--nper")+1]) if "--nper" in sys.argv else 1
    shuffle = not ('-noshuff' in sys.argv)
    smart = '-smart' in sys.argv

    auccess = []
    for eval_setup in ['ball_within_template', 'ball_cross_template']:
        auccess.append(eval_setup)
        for fold_id in range(3):
            if "-brute" in sys.argv:
                solver = FlownetSolver(model_path, "brute", smart=smart)
            else:
                solver = FlownetSolver(model_path, "pyramid", smart=smart)

            train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)

            if "-load" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| loading models...")
                solver.load_models(setup=eval_setup, fold=fold_id)

            if "-train" in sys.argv:
                if "-brute" in sys.argv:
                    print(model_path, eval_setup, fold_id, "|| loading data for brute training...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True, shuffle=shuffle)
                    print(model_path, eval_setup, fold_id, "|| training 'brute search' models...")
                    solver.train_brute_search(epochs=epochs)
                else:
                    print(model_path, eval_setup, fold_id, "|| loading data for generative training...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                    print(model_path, eval_setup, fold_id, "|| training 'generative' models...")
                    solver.train_supervised(epochs=epochs)

            if "-save" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| saving models...")
                solver.save_models(setup=eval_setup, fold=fold_id)

            if "-inspect" in sys.argv:
                if "-brute" in sys.argv:
                    print(model_path, eval_setup, fold_id, "|| loading data for brute testing...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True, shuffle=False, test=True)
                    print(model_path, eval_setup, fold_id, "|| inspecting brute performance...")
                    solver.inspect_brute_search(eval_setup, fold_id)
                else:
                    print(model_path, eval_setup, fold_id, "|| loading data for generative testing...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=False, test=True)
                    print(model_path, eval_setup, fold_id, "|| inspecting generative performance...")
                    solver.inspect_supervised(eval_setup, fold_id)
                    
            if "-solve" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| getting auccess...")
                if "-brute" in sys.argv:
                    local_auccess = solver.brute_auccess(test_ids+dev_ids)
                    #auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                    os.makedirs(f'result/solver/result/{solver.path}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {local_auccess} \nepochs: 50")
                    auccess.append(local_auccess)
                else:
                    local_auccess = solver.generative_auccess_old(test_ids+dev_ids)
                    #auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                    os.makedirs(f'result/solver/result/{solver.path}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {local_auccess} \nepochs: 50")
                    auccess.append(local_auccess)
                    """
                    auccess.append( get_auccess(solver, test_ids+dev_ids, solve_noise=True, save_tries=True) )
                    os.makedirs(f'result/solver/result/{solver.path}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {auccess[-1]} \nepochs: 50")
                    """

            """
            all_tasks = train_ids+dev_ids+test_ids
            filter_list = ['00000', '00001', '00002', '00003', '00004','00005', 
                            '00006','00007','00008','00009','00010','00011',
                            '00012','00013','00014','00015','00016','00017',
                            '00018','00019','00020','00021','00022','00023', '00024']
            """
            """
            auccess_per_task = dict()
            for filter_task in filter_list:
                selected_tasks = [t for t in all_tasks if t[:5]==filter_task]
                auccess = get_auccess(solver, selected_tasks, solve_noise=True, save_tries=True)
                auccess_per_task[filter_task] = auccess
            print(auccess_per_task)
            """

    print(auccess)