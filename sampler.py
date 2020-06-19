import phyre
import cv2
import pickle
import pathlib
import os
from PIL import Image
import numpy as np

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

def collect_observations(path, n_per_task = 10):
    tries = 0
    tasks = ['00000:001', '00000:002', '00000:003', '00000:004', '00000:005',
            '00001:001', '00001:002', '00001:003', '00001:004', '00001:005',
            '00002:007', '00002:011', '00002:015', '00002:017', '00002:023',
            '00003:000', '00003:001', '00003:002', '00003:003', '00003:004',
            '00004:063', '00004:071', '00004:092', '00004:094', '00004:095']
    base_path = path
    number_to_solve = n_per_task

    for task in tasks:
        sim = phyre.initialize_simulator([task], 'ball')
        solved = 0
        while solved < number_to_solve:
            tries += 1
            action = sim.sample()
            res = sim.simulate_action(0, action, 
                need_featurized_objects=True, stride=15)
            if res.status.is_solved():
                print("solved "+task+" with", tries, "tries")
                tries = 0
                solved += 1
                path_str = f"{base_path}/{task[:5]}/{task[6:]}/{str(solved)}"
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

def load_phyre_rollouts(path, flat=True, image=False):
    s = "/"
    for task in os.listdir(path):
        for variation in os.listdir(path+s+task):
            for trialfolder in os.listdir(path+s+task+s+variation):
                for fp in os.listdir(path+s+task+s+variation+s+trialfolder):
                    if fp=="observations.pickle":
                        final_path = path+s+task+s+variation+s+trialfolder+s+fp
                        #print(final_path)
                        if image:
                            yield(Image.open(final_path))
                        else:
                            with open(final_path, 'rb') as handle:
                                if flat:
                                    yield(pickle.load(handle))
                                else:
                                    yield(pickle.load(handle))
                    
if __name__ == "__main__":
    #collect_observations("phyre_obs")
    collect_all_observations("phyre_all_obs", n_per_task=1)
    #print(load_phyre_rollouts("phyre_obs"))
            