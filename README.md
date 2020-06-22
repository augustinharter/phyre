# PHYRE
Project space for tackling the PHYsical REasoning benchmark test (phyre.ai)

## PHYRE roullouts
use `sampler.py` to collect solving phyre rollouts from the original simulator

## Pymunk Simulator
For concept exploration and proofing pymunk is used to imitate the pyhre simulator.
Pymunk can be completely intercepted and customized to try out concepts and to gather custom data.

to generate raw rollouts with full scenes customize and run:  
`python action_rollouter.py [num_of_rollouts]`

to extract training data from raw scenes run:  
`python scene_extractor.py`

## Imaginet: Spatial CNN for solving path prediction
SCNN generates feasability map of where the goal ball can be found in a solving path, based on the inital scene (before any action was taken). This is further passed into another modul which is trained to predict the action path which could further be used to extract the actual action position (work in progress)  
From left to right: True Target Trajectory, Generated Target Trajectory, Base Trajectory without Action, True Action Trajectory, Generated Action Trajectory
![Results](/result/scnn/a-path/a-path-results.png)
