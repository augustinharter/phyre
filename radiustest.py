import phyre
from matplotlib import pyplot as plt
import os

sim = phyre.initialize_simulator(['00000:000'], 'ball')
for r in range(1, 11, 1):
  res = sim.simulate_action(0, [0.5,0.5,r*0.1], need_featurized_objects=True)
  print(res.featurized_objects.diameters)
  os.makedirs(f'result/radiustest/', exist_ok=True)
  plt.imsave(f'result/radiustest/{r*0.1}.png', res.images[0])