import phyre
from matplotlib import pyplot as plt
import os
import numpy as np

sim = phyre.initialize_simulator(['00000:000'], 'ball')
for r in range(1, 10, 1):
  res = sim.simulate_action(0, [r*0.1, r*0.1, 0.1], need_featurized_objects=True)
  if not res.status.is_invalid():
    print(res.featurized_objects.diameters)
    print(res.featurized_objects.features[0])
    os.makedirs(f'result/actiontest/', exist_ok=True)
    plt.imsave(f'result/actiontest/{r*0.1},{r*0.1}.png', np.flip(res.images[0], axis=0))