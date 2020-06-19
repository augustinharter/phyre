# PHYRE
Project space for tackling the PHYsical REasoning benchmark test (phyre.ai)

## Pymunk Simulator
For concept exploration and proofing pymunk is used to imitate the pyhre simulator.
Pymunk can be completely intercepted and customized to try out concepts and to gather custom data.

to generate raw rollouts with full scenes run:  
python action_rollouter.py [num_of_rollouts]

to extract training data from raw scenes run:  
python scene_extractor.py

## Imaginet: Spatial CNN for solving path prediction
SCNN generates feasability map of where the goal ball can be found in a solving path