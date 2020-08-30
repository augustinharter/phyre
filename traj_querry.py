import pickle
import numpy as np

with open('result/traj_lookup/lookup.pickle', 'rb') as fp:
    table = pickle.load(fp)

dummy_querry = (0,5)
dummy_traj = np.array((2.5, 2.1))

if dummy_querry in table:
    # Szenario/querry is known
    possible_trajectories = table[dummy_querry].keys()
    print(possible_trajectories)

    rounded_traj = tuple(np.round(dummy_traj))
    if rounded_traj in table[dummy_querry]:
        print(rounded_traj, 'exists', table[dummy_querry][rounded_traj], 'times')
