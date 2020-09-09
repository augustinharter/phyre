import pickle
import numpy as np

with open('result/traj_lookup/all_tasks/lookup.pickle', 'rb') as fp:
    table = pickle.load(fp)

print([(x,y) for x,y in table.keys() if y==0 and x>=0])
dummy_querry = (0,5)
dummy_traj = np.array((2.5, 2.1))

if dummy_querry in table:
    # Szenario/querry is known
    possible_trajectories = table[dummy_querry].keys()
    print(f"possible trajectoryis for dummy querry {dummy_querry}", possible_trajectories)

    rounded_traj = tuple(np.round(dummy_traj))
    if rounded_traj in table[dummy_querry]:
        print(rounded_traj, 'exists', table[dummy_querry][rounded_traj], 'times')
