import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
from phyre_rolllout_collector import load_phyre_rollouts, collect_solving_observations
import cv2
import phyre
import os
import pickle


def make_dual_dataset(path, size=(32,32), save=True):
    if os.path.exists(path+".pickle"):
        with open(path+'.pickle', 'rb') as fhandle:
            X, Y = pickle.load(fhandle)
    else:
        X = load_phyre_rollouts(path)
        X, Y = prepare_data(X, size)
        X = T.tensor(X).float()
        Y = T.tensor(Y).float()
        if save:
            with open(path+'.pickle', 'wb') as fhandle:
                pickle.dump((X,Y), fhandle)
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X,Y), 32, shuffle=True)
    return dataloader

def make_mono_dataset(path, size=(32,32), save=True, tasks=[]):
    if os.path.exists(path+".pickle"):
       X = T.load(path+'.pickle')
       index = T.load(path+'_index.pickle')
       print("Loaded dataset with shape:", X.shape)
    else:
        if tasks:
            collect_solving_observations(path, tasks, n_per_task=1, stride=5)
            pass
        data_generator = load_phyre_rollout_data(path)
        data, index = format_raw_rollout_data(data_generator, size=size)
        X = T.tensor(data).float()
        if save:
            T.save(X, path+'.pickle')
            T.save(index, path+'_index.pickle')
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), 32, shuffle=True)
    return dataloader, index

def prepare_data(data, size):
    targetchannel = 1
    X, Y = [], []
    print("Preparing dataset...")
    #x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for variations in data:
        with_base = len(variations) > 1
        for (j, rollout) in enumerate(variations):
            if not isinstance(rollout, np.ndarray):
                break
            #length = (2*len(rollout))//3
            #rollout = rollout[:length]
            roll = np.zeros((len(rollout), 7, size[0], size[1]))
            for i, scene in enumerate(rollout):
                channels = [(scene==j).astype(float) for j in range(1,8)]
                roll[i] = np.stack([(cv2.resize(c, size, cv2.INTER_MAX)>0).astype(float) for c in channels])
            roll = np.flip(roll, axis=2)
            trajectory = (np.sum(roll[:,targetchannel], axis=0)>0).astype(float)
            if not(with_base and j == 0):
                action = (np.sum(roll[:,0], axis=0)>0).astype(float)
            #goal_prior = dist_map(roll[0, 2] + roll[0, 3])
            #roll[0, 0] = goal_prior
            # TESTING ONLY
            #roll[0, 1] = roll[0, 0]
            if with_base and j == 0:
                base = trajectory
            else:
                action_ball = roll[0, 0].copy()
                roll[0, 0] = np.zeros_like(roll[0,0])
                #print(goal_prior)
                # Contains the initial scene without action
                X.append(roll[0])
                # Contains goaltarget, actiontarget, basetrajectory
                Y.append(np.stack((trajectory, action, base if with_base else np.zeros_like(roll[0,0]), action_ball)))
                #plt.imshow(trajectory)
                #plt.show()
    print("Finished preparing!")
    return X, Y

def extract_channels_and_paths(rollout, path_idxs=[1,0], size=(32,32), gamma=1):
    """
    returns init scenes from 'channels' followed by paths specified by 'path_idxs' 
    """
    paths = np.zeros((len(path_idxs), len(rollout), size[0], size[1]))
    alpha = 1
    for i, chans in enumerate(rollout):
        # extract color codings from channels
        #chans = np.array([(scene==ch).astype(float) for ch in channels])

        # if first frame extract init scene
        if not i:
            init_scene = np.array([(cv2.resize(chans[ch], size, cv2.INTER_MAX)>0).astype(float) for ch in range(len(chans))])

        # add path_idxs channels to paths
        for path_i, idx in enumerate(path_idxs):
            paths[path_i, i] = alpha*(cv2.resize(chans[idx], size, cv2.INTER_MAX)>0).astype(float)
        alpha *= gamma
    
    # flip y axis and concat init scene with paths
    paths = np.flip(np.max(paths, axis=1).astype(float), axis=1)
    init_scene = np.flip(init_scene, axis=1)
    result = np.concatenate([init_scene, paths])
    return result

def format_raw_rollout_data(data, size=(32,32)):
    targetchannel = 1
    data_bundle = []
    lib_dict = dict()
    print("Formating data...")
    #x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, (base, trial, info) in enumerate(data):
        print(f"at sample {i}; {info}")
        #base_path = extract_channels_and_paths(base, channels=[1], path_idxs=[0], size=size)[1]
        #trial_channels = extract_channels_and_paths(trial, size=size)
        #sample = np.append(trial_channels, base_path[None], axis=0)
        try:
            task, subtask, number = info
            base_path = extract_channels_and_paths(base, path_idxs=[1], size=size)[-1]
            trial_channels = extract_channels_and_paths(trial, path_idxs=[1,2,0], size=size)
            sample = np.append(trial_channels, base_path[None], axis=0)
            #plt.imshow(np.concatenate(tuple(np.concatenate((sub, T.ones(32,1)*0.5), axis=1) for sub in sample), axis=1))
            #plt.show()
            data_bundle.append(sample)
            
            # Create indexing dict
            key = task+':'+subtask
            if not key in lib_dict:
                lib_dict[key] = [i]
            else:
                lib_dict[key].append(i)
        except Exception as identifier:
            print(identifier)
    print("Finished preparing!")
    return data_bundle, lib_dict

def load_phyre_rollout_data(path, base=True):
    s = "/"
    fp ="observations.pickle"
    for task in os.listdir(path):
        for variation in os.listdir(path+s+task):
            if base:
                with open(path+s+task+s+variation+s+'base'+s+fp, 'rb') as handle:
                    base_rollout =  pickle.load(handle)
            for trialfolder in os.listdir(path+s+task+s+variation):                
                final_path = path+s+task+s+variation+s+trialfolder+s+fp
                with open(final_path, 'rb') as handle:
                    trial_rollout = pickle.load(handle)
                if base:
                    yield(base_rollout, trial_rollout, (task, variation, trialfolder))
                else:
                    yield(trial_rollout)
    
def pic_to_action_vector(pic, r_fac=3):
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y,x]:
                X += pic[y,x]*x
                Y += pic[y,x]*y
    summed = pic.sum()
    X /= pic.shape[0]*summed
    Y /= pic.shape[0]*summed
    R = np.sqrt(summed)/pic.shape[0]
    return X.item(), 1-Y.item(), r_fac*R.item()

def scenes_to_channels(X, size=(32,32)):
    x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, scene in enumerate(X):
        channels = [(scene==j).astype(float) for j in range(1,8)]
        x[i] = np.flip(np.stack([(cv2.resize(c, size, cv2.INTER_MAX)>0).astype(float) for c in channels]), axis=1)
    return x

def rollouts_to_specific_paths(batch, channel, size=(32,32), gamma=1):
    trajectory = np.zeros((len(batch), size[0], size[1]))
    for j, r in enumerate(batch):
        path = np.zeros((len(r), size[0], size[1]))
        alpha = 1
        for i, scene in enumerate(r):
            chan = (scene==channel).astype(float)
            path[i] = alpha*(cv2.resize(chan, size, cv2.INTER_MAX)>0).astype(float)
            alpha *= gamma
        path = np.flip(path, axis=1)
        base = np.max(path, axis=0).astype(float)
        trajectory[j] = base
    return trajectory


def visualize_actions_from_cache(amount):
    cache = phyre.get_default_100k_cache("ball")
    actions = cache.action_array[:amount]
    plt.scatter(actions[:,0], actions[:,1], alpha=0.3, s=1000*actions[:,2], c=actions[:,2])
    plt.show()

def print_folds():
    eval_setup = 'ball_within_template'
    for fold_id in range(10):
        #print(phyre.get_fold(eval_setup, fold_id)[0][:10])
        print(phyre.get_fold(eval_setup, fold_id)[0][:10] 
            == phyre.get_fold(eval_setup, fold_id)[0][:10])

def get_auccess_for_n_tries(n):
    eva = phyre.Evaluator(['00000:000'])
    for _ in range(n-1):
        eva.maybe_log_attempt(0, phyre.SimulationStatus.NOT_SOLVED)
    for _ in range(101-n):
        eva.maybe_log_attempt(0, phyre.SimulationStatus.SOLVED)
    return eva.get_auccess()

if __name__ == "__main__":
    #visualize_actions_from_cache(1000)
    #print(get_auccess_for_n_tries(10))
    print_folds()