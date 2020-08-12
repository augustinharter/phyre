#%%
from action_cgan import *
from phyre_utils import pic_to_action_vector
from phyre_rolllout_collector import collect_interactions
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools
from matplotlib import pyplot as plt

#%%
def solve(tasks, generator, save_images=False, force_collect=False, static=256, show=False):
    # Collect Interaction Data
    data_path = './data/cgan_solver'
    if not os.path.exists(data_path) or force_collect:
        os.makedirs(data_path)
        wid = generator.width
        collect_interactions(data_path, tasks, stride=1, size=(wid,wid), static=static)
    with open(data_path+'interactions.pickle', 'rb') as fs:
        X = T.tensor(pickle.load(fs), dtype=T.float)
    with open(data_path+'tasklist.pickle', 'rb') as fs:
        tasklist = pickle.load(fs)
    print('loaded dataset with shape:', X.shape)
    #data_set = T.utils.data.TensorDataset(X)
    #data_loader = T.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=False)

    # Sim SETUP
    print('Succesfull collection for tasks:\n', tasklist)
    eval_setup = 'ball_within_template'
    sim = phyre.initialize_simulator(tasklist, 'ball')
    eva = phyre.Evaluator(tasklist)

    # Solve Loop
    solved, tried = 0, 0
    for i,task in enumerate(tasklist):
        tried += 1
        fake = generator((X[i,generator.s_chan])[None])[0,0]
        action = np.array(pic_to_action_vector(fake))
        res = sim.simulate_action(i, action)

        # Noisy tries while invalid actions
        t = 0
        while res.is_invalid and t <100:
            t += 1
            action += np.random.rand(3)*0.01
            res = sim.simulate_action(i, action)

        # Check for and log Solves
        if res.status.is_solved():
            solved +=1
        print(task, "solved", res.status.is_solved())
        if show:
            plt.imshow(fake)
            plt.show()
            plt.imshow(res.images[0])
            plt.show()

    print("solving percentage:", solved/tried)

    """
    base_path = []
    action_path = []
    for i, t in enumerate(tasks):
        while True:
            action = sim.sample(i)
            action[2] = 0.01
            res = sim.simulate_action(i, action, stride=20)
            if type(res.images)!=type(None):
                base_path.append(rollouts_to_channel([res.images], 2))
                action_path.append(rollouts_to_channel([res.images], 1))
                break
    base_path = T.tensor(np.concatenate(base_path)).float()
    action_path = T.tensor(np.concatenate(base_path)).float()
    with T.no_grad():
        Z = model(X)
        A = model2(T.cat((X[:,1:], base_path[:,None], Z), dim=1))
    #B = model3(T.cat((X[:,1:], Y[:,None,2], Z, A), dim=1))
    #B = extract_action(A, inspect=-2 if save_images else -1)
    B = extract_action(action_path[:,None], inspect=-2 if save_images else -1)

    # Saving Images:
    if save_images:
        for inspect in range(len(X)):
            plt.imsave(f"result/flownet/{inspect}_init.png", T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[inspect]), dim=1))
            plt.imsave(f"result/flownet/{inspect}_base.png", base_path[inspect])
            plt.imsave(f"result/flownet/{inspect}_target.png", Z[inspect,0])  
            #plt.imsave(f"result/flownet/{inspect}_init_scene.png", np.flip(batch[inspect][0], axis=0))  
            plt.imsave(f"result/flownet/{inspect}_action.png", A[inspect,0])  
            plt.imsave(f"result/flownet/{inspect}_selection.png", B[inspect,0])
    gen_actions = []
    for b in B[:,0]:
        gen_actions.append(pic_to_values(b))
    print(gen_actions)
    
    # Feed actions into simulator
    eva = phyre.Evaluator(tasks)
    solved, valid, comb = dict(), dict(), dict()
    for i, t in enumerate(tasks):
        if not (t[:5] in comb):
            comb[t[:5]] = 0
            valid[t[:5]] = 0
            solved[t[:5]] = 0

        base_action = gen_actions[i]
        # Random Agent Intercept:
        #action = sim.sample()
        res = sim.simulate_action(i, base_action)
        tries = 0
        alpha = 1
        # 100 Tries Max:
        while eva.get_attempts_for_task(i)<100:
            if not res.status.is_solved():
                action = np.array(base_action)+np.random.randn(3)*np.array([0.1,0.1,0.1])*alpha
                res = sim.simulate_action(i, action)

                subtries = 0
                while subtries < 100 and res.status.is_invalid():
                    subtries += 1
                    action_var = np.array(action)+np.random.randn(3)*np.array([0.05,0.05,0.05])*alpha
                    res = sim.simulate_action(i, action_var)

                eva.maybe_log_attempt(i, res.status)
                alpha *=1.01
            else:
                eva.maybe_log_attempt(i, res.status)
            tries +=1

        if save_images:
            try:
                for k, img in enumerate(res.images):
                    plt.i#%%
        """

#%%
if __name__ == "__main__":
    wid = 64
    full = True
    generator = Generator(wid, 100, 4, 1, zoomed=not full)
    generator.load_state_dict(T.load("./saves/action_cgan/full-single/generator.pt"))

    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids+dev_ids+test_ids
    template13_tasks = [t for t in all_tasks if t.startswith('00013:')]
    template2_tasks = [t for t in all_tasks if t.startswith('00002:')]

    solve(template2_tasks, generator)
