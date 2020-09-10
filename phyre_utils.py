import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
from phyre_rolllout_collector import load_phyre_rollouts, collect_solving_observations, collect_solving_dataset
import cv2
import phyre
import os
import pickle
import random
import json
from PIL import ImageDraw, Image, ImageFont


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

def make_mono_dataset_old(path, size=(32,32), save=True, tasks=[], shuffle=True):
    if os.path.exists(path+".pickle") and os.path.exists(path+"_index.pickle"):
       X = T.load(path+'.pickle')
       index = T.load(path+'_index.pickle')
       print(f"Loaded dataset from {path} with shape:", X.shape)
    else:
        if tasks:
            collect_solving_observations(path, tasks, n_per_task=1, stride=5, size=size)
        data_generator = load_phyre_rollout_data(path)
        data, index = format_raw_rollout_data(data_generator, size=size)
        X = T.tensor(data).float()
        if save:
            T.save(X, path+'.pickle')
            T.save(index, path+'_index.pickle')
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), 32, shuffle=shuffle)
    return dataloader, index

def make_mono_dataset(path, size=(32,32), tasks=[], batch_size = 32, solving=True, n_per_task=1, shuffle=True):
    if os.path.exists(path+"/data.pickle") and os.path.exists(path+"/index.pickle"):
        with open(path+'/data.pickle', 'rb') as fp:
            data = pickle.load(fp)
            X = T.tensor(data).float()
        with open(path+'/index.pickle', 'rb') as fp:
            index = pickle.load(fp)
        print(f"Loaded dataset from {path} with shape:", X.shape)
    else:
        if tasks:
            collect_solving_dataset(path, tasks, n_per_task=n_per_task, stride=5, size=size, solving=solving)
        with open(path+'/data.pickle', 'rb') as fp:
            data = pickle.load(fp)
        with open(path+'/index.pickle', 'rb') as fp:
            index = pickle.load(fp)
        X = T.tensor(data).float()
        print(f"Loaded dataset from {path} with shape:", X.shape)
        
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), batch_size, shuffle=shuffle)
    return dataloader, index

def vis_batch(batch, path, pic_id, text = []):
    #print(batch.shape)

    if len(batch.shape) == 4:
        padded = F.pad(batch, (1,1,1,1), value=0.5)
    elif len(batch.shape) == 5:
        padded = F.pad(batch, (0,0,1,1,1,1), value=0.5)
    else:
        print("Unknown shape:", batch.shape)

    #print(padded.shape)
    reshaped = T.cat([T.cat([channels for channels in sample], dim=1) for sample in padded], dim=0)
    #print(reshaped.shape)
    if np.max(reshaped.numpy())>1.0:
        reshaped = reshaped/256
    os.makedirs(path, exist_ok=True)
    if text:
        text_height= 30
        if len(reshaped.shape) == 2:
            reshaped = F.pad(reshaped, (0,0,text_height,0), value=0.0)
            img = Image.fromarray(np.uint8(reshaped.numpy()*255), mode="L")
        elif len(reshaped.shape) == 3:
            reshaped = F.pad(reshaped, (0,0,0,0,text_height,0), value=0.0)
            img = Image.fromarray(np.uint8(reshaped.numpy()*255))
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 9)
        draw = ImageDraw.Draw(img)
        for i, words in enumerate(text):
            x, y = i*reshaped.shape[1]//len(text), 0
            draw.text((x, y), words, fill=(255) if len(reshaped.shape)==2 else (255,255,255), font=font)
        img.save(f'{path}/'+pic_id+'.png')
    else:
        plt.imsave(f'{path}/'+pic_id+'.png', reshaped.numpy(), dpi=1000)

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
    
def draw_ball(w, x, y, r, invert_y=False):
        """inverts y axis """
        x = int(w*x)
        y = int(w*(1-y)) if invert_y else int(w*y)
        r = w*r
        X = T.arange(w).repeat((w, 1)).float()
        Y = T.arange(w).repeat((w, 1)).transpose(0, 1).float()
        X -= x # X Distance
        Y -= y # Y Distance
        dist = (X.pow(2)+Y.pow(2)).pow(0.5)
        return (dist<r).float()

def action_delta_generator():
    temp = 1
    radfac = 0.025
    coordfac = 0.07
    for fac in [1,2]:
        for rad in [0,1,-1]:
            for xd,yd in [(1,0), (-1,0), (2,0), (-2,0), (-1,2), (1,2), (-1,-2), (-1,-2)]:
                #print((fac*np.array((coordfac*xd, coordfac*yd, rad*radfac))))
                yield (fac*np.array((coordfac*xd, coordfac*yd, rad*radfac)))
    
    count = 0
    while True:
        count += 1
        action = ((2*np.random.rand(3)-1)*np.array([0.2,0.1,0.05])*temp)
        #print(count,"th", "ACTION:", action)
        yield np.abs(action)
        temp = 1.04*temp if temp<3 else temp

def pic_to_action_vector(pic, r_fac=1):
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y,x]:
                X += pic[y,x]*x
                Y += pic[y,x]*y
    summed = pic.sum()
    X /= pic.shape[0]*summed
    Y /= pic.shape[0]*summed
    r = np.sqrt(pic.sum()/(3.141592*pic.shape[0]**2))
    return [X.item(), 1-Y.item(), r_fac*r.item()]

def grow_action_vector(pic, r_fac=1, show=False):
    id = int((T.rand(1)*100))
    os.makedirs("result/flownet/solver/grower", exist_ok=True)
    plt.imsave(f"result/flownet/solver/grower/{id}.png", pic)
    pic = pic*(pic>pic.mean())
    plt.imsave(f"result/flownet/solver/grower/{id}_thresh.png", pic)


    wid = pic.shape[0]

    def get_value(x,y,r):
        ball = draw_ball(wid, x, y, r)
        potential = T.sum(ball)
        actual = T.sum(pic[ball.bool()])
        value =  (actual**0.5)*actual/potential
        return value

    def move_and_grow(x,y,r,v):
        delta = 0.7
        positions = [(x+dx,y+dy) for (dx,dy) in [(-(0.3+delta)/wid,0), ((0.3+delta)/wid,0), (0,-(0.3+delta)/wid), (0,(0.3+delta)/wid)] if (0<=x+dx<1) and (0<=y+dy<1)]
        bestpos = (x,y)
        bestrad = r
        bestv = v
        for pos in positions:
            value = get_value(*pos, r)
            rad, val = grow(*pos,r,value)
            if val>bestv:
                bestpos = pos
                bestrad = rad
                bestv = val
        return bestpos[0], bestpos[1], bestrad, bestv

    def grow(x,y,r,v):
        bestv = v
        bestrad = r
        for rad in [r+0.3/wid, r+0.6/wid, r+0.9/wid, r-0.35/wid]:
            if 0<rad<wid:
                value = get_value(x,y,rad)
                if value>bestv:
                    bestv = value
                    bestrad = rad
        return bestrad, bestv

    seeds = []
    num_seeds = 5
    while len(seeds)<num_seeds:
        r = 3/32
        try:
            y, x = random.choice(np.nonzero(pic>0.1))
            seeds.append((x.item()/wid,y.item()/wid,r))
        except:
            y, x = wid//2, wid//2
            seeds.append((x/wid,y/wid,r))

    final_seeds = []
    for (x,y,r) in seeds:
        v = get_value(x,y,r)
        #plt.imshow(pic+draw_ball(wid,x,y,r))
        #plt.show()
        for i in range(10):
            x, y, r, v = move_and_grow(x,y,r,v)
            #r, v = grow(x,y,r,v)
            if show:
                plt.imshow(pic+draw_ball(wid,x,y,r))
                plt.show()
        final_seeds.append(((x,y,r),v))
        
    action = np.array(max(final_seeds, key= lambda x: x[1])[0])
    action[1] = 1-action[1]
    plt.imsave(f"result/flownet/solver/grower/{id}_drawn.png", draw_ball(wid, *action, invert_y = True))
    action[2]*=r_fac
    compare_action = action.copy()
    action[2] = action[2] if action[2]<0.25 else 0.24999
    action[0] = action[0] if action[0]> 0 else  0
    action[0] = action[0] if action[0]<1- 0 else 1- 0
    action[1] = action[1] if action[1]> 0 else  0
    action[1] = action[1] if action[1]<1- 0 else 1- 0
    if np.any(action!=compare_action):
        print("something was out of bounce:", action, compare_action)
    return action

def pic_hist_to_action(pic, r_fac=3):
    # thresholding
    pic = pic*(pic>0.2)
    # columns part of ball
    cols = [idx for (idx,val) in enumerate(np.sum(pic, axis=0)) if val>2]
    start, end = min(cols), max(cols)
    x = (start+end)/2
    x /= pic.shape[1]
    # rows part of ball
    rows = [idx for (idx,val) in enumerate(np.sum(pic, axis=1)) if val>2]
    start, end = min(rows), max(rows)
    y = (start+end)/2
    y /= pic.shape[0]
    # radius
    r = np.sqrt(pic.sum()/(3.141592*pic.shape[0]**2))
    r = 0.1
    return x, y, r

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

def collect_traj_lookup(tasks, save_path, number_per_task, show=False, stride=10):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    print("Amount per task", number_per_task)

    keys = []
    values = []

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task)==1]
            print(f"collecting {n_collected+1} interactions from {task} with {tries} tries", end = end_char)
            if len(action)==0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                need_featurized_objects=True, stride=1)
            while res.status.is_invalid():
                action = np.random.rand(3)
                res = sim.simulate_action(idx, action,
                    need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                #print(res.images.shape)
                #print(len(res.bitmap_seq))
                #print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                #print(idx1, idx2)
                #print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx,red_idx]])/2
                for i,m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        pos = res.featurized_objects.features[i,[green_idx,red_idx],:2]
                        dist = np.linalg.norm(pos[1]-pos[0])
                        #print(dist, target_dist)
                        if not dist<target_dist+0.005:
                            continue

                        red_radius = res.featurized_objects.diameters[red_idx]*4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i, pos[0], action_at_interaction, target_dist)

                return (False, 0, (0,0), 0, 0)

            contact, i_step, green_pos, red_pos, summed_radii = check_contact(res)
            if  contact:
                tries = 0

                step_n = 10
                # check whether contact happend too early
                if i_step-step_n < 0:
                    continue

                try:
                    green_idx = res.featurized_objects.colors.index('GREEN')
                    red_idx = res.featurized_objects.colors.index('RED')
                    green_minus, _ = res.featurized_objects.features[i_step-stride,[green_idx,red_idx],:2]
                    green_zero, _ = res.featurized_objects.features[i_step,[green_idx,red_idx],:2]
                    green_plus, _ = res.featurized_objects.features[i_step+stride,[green_idx,red_idx],:2]
                    green_key, _ = green_minus-green_zero, 0
                    green_value, _ = green_zero-green_plus, 0
                    keys.append((green_key[0], green_key[1]))
                    values.append((green_value[0], green_value[1]))
                except:
                    continue
                
                n_collected += 1

            if tries>max_tries:
                break

    keys = np.round(256*np.array(keys))
    k_x_max = keys[np.argmax(np.abs(keys[:,0])),0]
    k_y_max = keys[np.argmax(np.abs(keys[:,1])),1]
    """keys[:,0] /= k_x_max/5
    keys[:,1] /= k_y_max/5
    k_x_max = np.max(np.abs(keys[:,0]))
    k_y_max = np.max(np.abs(keys[:,1]))"""
    values = np.round(256*np.array(values))
    v_x_max = values[np.argmax(np.abs(values[:,0])), 0]
    v_y_max =  values[np.argmax(np.abs(values[:,1])), 1]
    """values[:,0] /= v_x_max/5
    values[:,1] /= v_y_max/5
    v_x_max = np.max(np.abs(values[:,0]))
    v_y_max = np.max(np.abs(values[:,1]))"""

    table = dict()
    for i in range(len(keys)):
        k = tuple(keys[i])
        v = tuple(values[i])
        if k in table:   
            table[k][v] = table[k][v] + 1 if v in table[k] else 1
        else:
            table[k] = {v:1}


    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/lookup.pickle', 'wb') as fp:
        pickle.dump(table, fp)
    print(f"FINISH collecting trajectory lookup!")
    return keys, values, k_x_max, k_y_max, v_x_max, v_y_max, table

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
    
    # Collecting trajectory lookup
    #pic = draw_ball(32,0.5,0.2,0.2) + draw_ball(32,0.5,0.4,0.1)
    #print(grow_action_vector(pic))
    #exit()
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids+dev_ids+test_ids
    template13_tasks = [t for t in all_tasks if t.startswith('00013:')]
    template2_tasks = [t for t in all_tasks if t.startswith('00002:')]
    print(template2_tasks)
    #collect_specific_channel_paths(f'./data/template13_action_paths_10x', template13_tasks, 0)
    keys, values, kxm, kym, vxm, vym, table = collect_traj_lookup(all_tasks, 'result/traj_lookup/all_tasks', 10, stride=10)
    print(keys)
    print(values)
    print(kxm, kym, vxm, vym)
    print(table)
    
