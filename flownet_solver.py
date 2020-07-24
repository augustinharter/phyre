#%%
from imaginet import *
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools

#%%
def solve(model, model2, save_images=False):
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
            '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
            '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
            '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
            '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']
    tasks = json.load(open("most_tasks.txt", 'r'))

    eval_setup = 'ball_within_template'
    fold_id = 0  # For simplicity, we will just use one fold for evaluation.
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    print('Size of resulting splits:\n train:', len(train_tasks), '\n dev:',
      len(dev_tasks), '\n test:', len(test_tasks))

    tasks = train_tasks[:]
    print("tasks:\n",tasks)
    sim = phyre.initialize_simulator(tasks, 'ball')
    init_scenes = sim.initial_scenes
    X = T.tensor(format(init_scenes)).float()
    print("Init Scenes Shape:\n",X.shape)

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
    spacing = [1,2,3,4]
    fig, ax = plt.subplots(5,5, sharey=True, sharex=True)
    for i, t in enumerate(comb):
        ax[i//5,i%5].bar(spacing, [solved[t[:5]]/(valid[t[:5]] if valid[t[:5]] else 1), solved[t[:5]]/comb[t[:5]], valid[t[:5]]/comb[t[:5]], comb[t[:5]]/100])
        ax[i//5,i%5].set_xlabel(t[:5])
    plt.show()



#%%
if __name__ == "__main__":
    """
    img = cv2.imread("result/scnn/a-path/1_actionmap.png", cv2.IMREAD_GRAYSCALE)
    X = T.tensor(img)[None,None,:].float()/255
    print(X.shape)
    X = (X*(X>0.2))
    plt.imshow(X[0,0])
    plt.show()
    X = extract_action(X, inspect=-2)
    plt.imsave("result/flownet/0_selection.png", X[0,0])
    print(pic_to_values(X[0,0]))
    """
    model = FlowNet(7, 16)
    model2 = FlowNet(8, 16)
    model.load_state_dict(T.load("saves/imaginet-c16-all.pt"))
    model2.load_state_dict(T.load("saves/imaginet2-c16-all.pt"))
    solve(model, model2, save_images=False)


# %%
