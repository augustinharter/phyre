import phyre
import cv2
import pickle
import pathlib
import os
from PIL import Image
import numpy as np
import json
import random
import cv2
import matplotlib.pyplot as plt
from planner.planner_agent import solve

def collect_images():
    tries = 0
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
            '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
            '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
            '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
            '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']

    tasks = ["00019:612"]
    base_path = "fiddeling"
    number_to_solve = 20

    for task in tasks:
        sim = phyre.initialize_simulator([task], 'ball')
        solved = 0
        while solved < number_to_solve:
            tries += 1
            action = sim.sample()
            res = sim.simulate_action(0, action, need_featurized_objects=True)
            if res.status.is_solved():
                print("solved "+task+" with", tries, "tries")
                tries = 0
                solved += 1
                #print(res.images.shape)
                for i, scene in enumerate(res.images):
                    img = phyre.observations_to_uint8_rgb(scene)
                    path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
                    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(path_str+f"/{str(i)}.jpg", img)
                    with open(path_str+"/objects.pickle", 'wb') as handle:
                        pickle.dump(res.featurized_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        #print(res.featurized_objects)
                    with open(path_str+"/action.txt", 'w') as handle:
                        handle.write(str(action))

def collect_solving_observations(path, tasks, n_per_task = 10, collect_base=True, stride=10, size=(32,32)):
    end_char = '\r'
    tries = 0
    max_tries = 100
    base_path = path
    number_to_solve = n_per_task
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    lib_dict = dict()

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        solved = 0
        while solved < number_to_solve:
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
            if not os.path.exists(path_str+"/observations.pickle"):
                print(f"collecting {task}: trial {solved} with {tries+1} tries", end = end_char)
                tries += 1
                action = actions[cache.load_simulation_states(task)==1]
                if len(action)==0:
                    print("no solution action in cache at task", task)
                    action = [np.random.rand(3)]
                action = random.choice(action)
                res = sim.simulate_action(idx, action,
                    need_featurized_objects=True, stride=stride)
                if res.status.is_solved():
                    tries = 0
                    solved += 1
                    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                    with open(path_str+"/observations.pickle", 'wb') as handle:
                        rollout = np.array([[cv2.resize((scene==ch).astype(float), size, cv2.INTER_MAX) for ch in range(1,7)] for scene in res.images])
                        pickle.dump(rollout, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if tries>max_tries:
                    break
            else:
                solved += 1
                print(f"skipping {task}: trial {solved}", end = end_char)
        
        # COLLECT BASE
        if collect_base and tries<=max_tries:
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/base"
            if not os.path.exists(path_str+"/observations.pickle"):
                print(f"collecting {task}: base", end= end_char)
                # 10 tries to make increase chance of one action being valid
                for _ in range(10):
                    action = sim.sample()
                    action[2] = 0.01
                    res = sim.simulate_action(idx, action, 
                        need_featurized_objects=True, stride=stride)
                    if not res.status.is_invalid():
                        break
                pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                with open(path_str+"/observations.pickle", 'wb') as handle:
                    rollout = np.array([[cv2.resize((scene==ch).astype(float), size, cv2.INTER_LINEAR) for ch in range(1,7)] for scene in res.images])
                    pickle.dump(rollout, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f"skipping {task}: base", end = end_char)

    print("FINISH collecting rollouts!")

def collect_specific_channel_paths(path, tasks, channel, stride=10, size=(256,256)):
    end_char = '\r'
    tries = 0
    max_tries = 100
    base_path = path
    number_to_solve = 10
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    data_dict = dict()

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        solved = 0
        while solved < number_to_solve:
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
            if not os.path.exists(path_str+"/observations.pickle"):
                print(f"collecting channel {channel} from {task}: trial {solved} with {tries+1} tries", end = end_char)
                tries += 1
                action = actions[cache.load_simulation_states(task)==1]
                if len(action)==0:
                    print("no solution action in cache at task", task)
                    action = [np.random.rand(3)]
                action = random.choice(action)
                res = sim.simulate_action(idx, action,
                    need_featurized_objects=True, stride=stride)
                if res.status.is_solved():
                    tries = 0
                    solved += 1
                    #pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                    rollout = np.array([[cv2.resize((scene==ch).astype(float), size, cv2.INTER_MAX) for ch in range(1,7)] for scene in res.images])
                    extracted_path = np.max(rollout[:,channel], axis=0)
                    # Collect index
                    key = task
                    if key in data_dict:
                        data_dict[key].append(extracted_path)
                    else:
                        data_dict[key] = [extracted_path]
                if tries>max_tries:
                    break
            else:
                solved += 1
                print(f"skipping {task}: trial {solved}", end = end_char)

    # Save data_dict
    with open(f'{base_path}/channel_paths.pickle', 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting channel {channel} paths!")

def collect_interactions(save_path, tasks, number_per_task, stride=1, size=(64,64), zoom=False, show=False):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    #base_path = 'data/fiddeling'
    data = []
    print("NUMBER", number_per_task)

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


                        #print(res.featurized_objects.diameters[[green_idx,red_idx]])
                        #print(res.featurized_objects.features[i,green_idx])
                        #print(res.featurized_objects.features[i, red_idx])
                        #print(i+2)
                        #for i, scene in enumerate(res.images):
                        #    img = phyre.observations_to_uint8_rgb(scene)
                        #    path_str = f"{base_path}/{task[:5]}/{task[6:]}"
                        #    pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                        #    cv2.imwrite(path_str+f"/{str(i)}.jpg", img[:,:,::-1])
                        

                        return (True, i, pos[0], target_dist)

                return (False, 0, (0,0), 0)

            contact, i_step, obj_pos, summed_radii = check_contact(res)
            if  contact:
                tries = 0
                n_collected += 1

                # setting up parameters for cutting out selection
                width = round(256*summed_radii*4)
                wh = width//2
                starty = round((obj_pos[1])*256)
                startx = round(obj_pos[0]*256)
                step_size = 20
                # check whether contact happend too early
                if i_step-step_size < 0:
                    continue

                selected_rollout = np.array([[(scene==ch).astype(float) for ch in range(1,7)] for scene in res.images[i_step-step_size:i_step+step_size+1:step_size]])
                #selected_rollout = np.flip(selected_rollout, axis=2)
                #print(selected_rollout.shape)
                ##padded_selected_rollout = np.pad(selected_rollout, ((0,0), (0,0), (wh,wh), (wh,wh)))
                #print(padded_selected_rollout.shape)
                ##extracted_scene = padded_selected_rollout[:,:,starty:starty+width, startx:startx+width]
                extracted_scene = np.flip(selected_rollout, axis=2)

                es = extracted_scene
                channel_formatted_scene = np.stack((es[0,1], es[1,1], es[2,1], np.max(es[1,2:], axis=0), es[0,0], es[1,0]))
                size_formatted_scene = [cv2.resize(img, size, cv2.INTER_MAX) for img in channel_formatted_scene]

                # saving extracted scene
                data.append(size_formatted_scene)

                if show:
                    plt.imshow(phyre.observations_to_uint8_rgb(res.images[i_step]))
                    plt.show()
                    fig, ax = plt.subplots(1,6)
                    for i,img in enumerate(channel_formatted_scene):
                        ax[i].imshow(img)
                    #plt.imshow(np.concatenate([*channel_formatted_scene], axis=1))
                    plt.show()
                    fig, ax = plt.subplots(1,6)
                    for i,img in enumerate(size_formatted_scene):
                        ax[i].imshow(img)
                    #plt.imshow(np.concatenate([*size_formatted_scene], axis=1))
                    plt.show()

            if tries>max_tries:
                break

    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/scene_interactions.pickle', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"FINISH collecting interactions!")

def collect_gridded_observations(path, n_per_task = 10):
    tries = 0
    tasks = ['00012:002', '00011:004', '00008:062', '00002:047']
    base_path = path
    number_to_solve = n_per_task

    for task in tasks:
        sim = phyre.initialize_simulator([task], 'ball')
        r = 0.2
        # Gridding:
        for (x,y) in [(x,y) for x in np.linspace(0.1,0.9,10) for y in np.linspace(0.1,0.9,10)]:
            tries = 0
            while tries < 20:
                tries += 1
                action = [x+(np.random.rand()-0.5)*0.1, y+(np.random.rand()-0.5)*0.1, r]
                res = sim.simulate_action(0, action, 
                    need_featurized_objects=True, stride=15)
                if not res.status.is_invalid():
                    break
            if res.status.is_invalid():
                continue
            path_str = f"{base_path}/{task[:5]}/{task[6:]}/{x}_{y}_{r}"
            pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
            with open(path_str+"/observations.pickle", 'wb') as handle:
                pickle.dump(res.images, handle, protocol=pickle.HIGHEST_PROTOCOL)

def collect_all_observations(path, n_per_task = 10):
    tries = 0
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
            '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
            '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
            '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
            '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']
    tasks = [f'000{"0"+str(t) if t<10 else t}:0{"0"+str(v) if v<10 else v}' for t in range(2, 100) for v in range(100)]
    #tasks = ['00000:001']

    base_path = path
    number_to_solve = n_per_task

    for task in tasks:
        print("trying", task)
        try:
            sim = phyre.initialize_simulator([task], 'ball')
        except Exception:
            continue
        solved = 0
        while solved < number_to_solve:
            tries += 1
            action = sim.sample()
            res = sim.simulate_action(0, action, 
                need_featurized_objects=True, stride=20)
            if res.status.is_solved():
                print("solved "+task+" with", tries, "tries")
                tries = 0
                solved += 1
                path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
                pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
                with open(path_str+"/observations.pickle", 'wb') as handle:
                    pickle.dump(res.images, handle, protocol=pickle.HIGHEST_PROTOCOL)

def collect_base_observations(path):
    tries = 0
    tasks = [f'000{"0"+str(t) if t<10 else t}:0{"0"+str(v) if v<10 else v}' for t in range(0, 25) for v in range(100)]
    #tasks = ['00000:001']
    base_path = path

    for task in tasks:
        print("trying", task)
        try:
            sim = phyre.initialize_simulator([task], 'ball')
        except Exception:
            continue
        print("running", task)
        action = sim.sample()
        action[2] = 0.01
        res = sim.simulate_action(0, action, 
            need_featurized_objects=True, stride=20)
        path_str = f"{base_path}/{task[:5]}/{task[6:]}/base"
        pathlib.Path(path_str).mkdir(parents=True, exist_ok=True)
        with open(path_str+"/observations.pickle", 'wb') as handle:
            pickle.dump(res.images, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_available_tasks():
    tasks = [f'000{"0"+str(t) if t<10 else t}:0{"0"+str(v) if v<10 else v}' for t in range(0, 25) for v in range(100)]
    #tasks = ['00000:001']

    available_tasks = []

    for task in tasks:
        print("trying", task)
        try:
            sim = phyre.initialize_simulator([task], 'ball')
            available_tasks.append(task)
        except Exception:
            continue
    print(available_tasks)
    json.dump(available_tasks, open("most_tasks.txt", 'w'))
        


def load_phyre_rollouts(path, image=False, base=True):
    s = "/"
    for task in os.listdir(path):
        for variation in os.listdir(path+s+task):
            tmp = []
            for trialfolder in os.listdir(path+s+task+s+variation):
                for fp in os.listdir(path+s+task+s+variation+s+trialfolder):
                    if fp=="observations.pickle":
                        final_path = path+s+task+s+variation+s+trialfolder+s+fp
                        if image:
                            yield(Image.open(final_path))
                        else:
                            with open(final_path, 'rb') as handle:
                                if base and trialfolder == "base":
                                    tmp.insert(0, pickle.load(handle))
                                else:
                                    tmp.append(pickle.load(handle))
            yield(tmp)
                    
if __name__ == "__main__":
    #collect_observations("phyre_obs")
    #collect_all_observations("data/phyre_test_obs", n_per_task=1)
    #collect_base_observations("data/phyre_all_obs")
    #print(np.array(list(load_phyre_rollouts("data/phyre_test_obs"))[:10]))
    #get_available_tasks()
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids+dev_ids+test_ids
    template13_tasks = [t for t in all_tasks if t.startswith('00013:')]
    template2_tasks = [t for t in all_tasks if t.startswith('00002:')]
    print(template2_tasks)
    #collect_specific_channel_paths(f'./data/template13_action_paths_10x', template13_tasks, 0)
    collect_interactions(f'./data/template2_interactions', template2_tasks, 50, 1, (64,64), show=False)