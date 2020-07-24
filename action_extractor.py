import torch as T
import numpy as np
import phyre
from phyre_utils import *
import random

def extract_action_map(X, inspect=-1):
    floor = 0.15
    top = 1.0
    # Creating Height mask which multiples top rows with 1 
    # and reduces factor towards floor for bottom row
    height_mask =  T.linspace(top, floor, X.shape[2]).repeat(X.shape[0], X.shape[3], 1).transpose(1,2)[:,None,:,:]
    x = F.avg_pool2d(X, 5, 1, 2)*height_mask
    if inspect >=0:
        plt.imshow(x[inspect,0])
        plt.show()
    if inspect == -2:
        for i in range(len(X)):
            plt.imsave(f"result/flownet/{i}_smooth.png", x[i,0])
    tmp = x.reshape(X.shape[0], X.shape[1], -1)
    args = tmp.argmax(dim=2)
    local_masks = T.zeros_like(X)
    points = T.zeros_like(tmp)
    for i in range(X.shape[0]):
        #tmp[i,0, args[i]] = 1
        points[i,0, args[i]] = 1
        j = args[i]//X.shape[3]
        k = args[i]%X.shape[3]
        local_masks[i,:,j-6 if j-6>0 else 0:j+7,k-6 if k-6>0 else 0:k+7] = 1
    if inspect >=0:
        plt.imshow(local_masks[inspect,0])
        plt.show()
    if inspect == -2:
        for i in range(len(X)):
            plt.imsave(f"result/flownet/{i}_mask.png", local_masks[i,0])
    #x = tmp.reshape_as(X)
    return X*local_masks+points.reshape_as(X)

if __name__ == "__main__":
    ## TESTING HANDCRAFTED ACTION EXTRACTOR WITH GROUNDTRUTH ACTION PATH
    # SETUP of phyre simulator
    SAVE_IMAGES = False
    eval_setup = 'ball_within_template'
    fold_id = 0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    cache = phyre.get_default_100k_cache("ball")
    actions = cache.action_array
    print(cache.task_ids)
    tasks = train_tasks#+dev_tasks+test_tasks
    print(f"{len(tasks)} tasks")
    sim = phyre.initialize_simulator(tasks, 'ball')
    init_scenes = sim.initial_scenes
    X = T.tensor(scenes_to_channels(init_scenes)).float()
    print("Init Scenes Shape:\n",X.shape)

    # COLLECT action path
    action_paths = []
    for i, t in enumerate(tasks):
        while True:
            action = actions[cache.load_simulation_states(t)==1]
            if len(action) == 0:
                action= [sim.sample()]
            action = random.choice(action)
            res = sim.simulate_action(i, action, stride=20)
            print(i, res.status.is_solved(), len(res.images), end='\r')
            if type(res.images)!=type(None):
                action_paths.append(rollouts_to_specific_paths([res.images], 1, gamma=0.9))
                break
    action_paths = T.tensor(np.concatenate(action_paths)).float()
    B = extract_action_map(action_paths[:,None], inspect=-2 if SAVE_IMAGES else -1)

    # Saving Images:
    if SAVE_IMAGES:
        for inspect in range(len(X)):
            plt.imsave(f"result/flownet/{inspect}_init.png", T.cat(tuple(T.cat((sub, T.ones(32,1)*0.5), dim=1) for sub in X[inspect]), dim=1))
            #plt.imsave(f"result/flownet/{inspect}_init_scene.png", np.flip(batch[inspect][0], axis=0))  
            plt.imsave(f"result/flownet/{inspect}_action.png", action_paths[inspect,0])  
            plt.imsave(f"result/flownet/{inspect}_selection.png", B[inspect,0])

    gen_actions = []
    for b in B[:,0]:
        gen_actions.append(pic_to_action_vector(b))
    print("Extracted actions:\n", gen_actions)
    
    # Feed actions into simulator
    eva = phyre.Evaluator(tasks)
    solved, valid, comb, avg_tries = dict(), dict(), dict(), dict()
    for i, t in enumerate(tasks):
        print(f"{i} solving {t}", end='\r')
        if not (t[:5] in comb):
            comb[t[:5]] = 0
            valid[t[:5]] = 0
            solved[t[:5]] = 0
            avg_tries[t[:5]] = []
        base_action = gen_actions[i]
        # Random Agent Intercept:
        #action = sim.sample()
        res = sim.simulate_action(i, base_action)
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
                if res.status.is_solved():
                    avg_tries[t[:5]].append(eva.get_attempts_for_task(i))
                eva.maybe_log_attempt(i, res.status)
                alpha *=1.01
            else:
                eva.maybe_log_attempt(i, res.status)

        if SAVE_IMAGES:
            try:
                for k, img in enumerate(res.images):
                    plt.imsave(f"result/flownet/{i}_{k}.png", np.flip(img, axis=0))
                    pass
            except Exception:
                pass
        #print(i, t, res.status.is_solved(), not res.status.is_invalid())
        comb[t[:5]] = comb[t[:5]]+1
        if not res.status.is_invalid():
            valid[t[:5]] = valid[t[:5]]+1
        if res.status.is_solved():
            solved[t[:5]] = solved[t[:5]]+1

    # Prepare Plotting
    print(eva.compute_all_metrics())
    print(eva.get_auccess())
    tries_per_task = [sum(avg_tries[k])/len(avg_tries[k]) for k in avg_tries]
    print("tries, per task", tries_per_task)
    auccess_per_task = list(enumerate(get_auccess_for_n_tries(round(t)) for t in tries_per_task))
    print("auccess per task", auccess_per_task)
    print("averaged auccess per tasks:", sum(v for _,v in auccess_per_task)/len(auccess_per_task))
    spacing = [1,2,3,4]
    fig, ax = plt.subplots(5,5, sharey=True, sharex=True)
    for i, t in enumerate(comb):
        ax[i//5,i%5].bar(spacing, [solved[t[:5]]/(valid[t[:5]] if valid[t[:5]] else 1), solved[t[:5]]/comb[t[:5]], valid[t[:5]]/comb[t[:5]], comb[t[:5]]/100])
        ax[i//5,i%5].set_xlabel(t[:5])
    plt.show()