#!/bin/sh

#conda init bash
#conda activate phyre

python action_cgan.py --path 3conv64-128 --data centered_64-128 --width 64 --epochs 300 -single --folds 3 --saveevery 30
python action_cgan.py --path 3conv64-128 --data centered_64-128 -genonly --width 64 -single --folds 3

python action_cgan.py --path 2conv64-128 --data centered_64-128 --width 64 --epochs 300 -single --folds 2 --saveevery 30
python action_cgan.py --path 2conv64-128 --data centered_64-128 -genonly --width 64 -single --folds 2

python action_cgan.py --path 1conv64-128 --data centered_64-128 --width 64 --epochs 300 -single --folds 1 --saveevery 30
python action_cgan.py --path 1conv64-128 --data centered_64-128 -genonly --width 64 -single --folds 1

python action_cgan.py --path linlin64-128 --data centered_64-128 --width 64 --epochs 300 -single -lindisc -lingen --saveevery 30
python action_cgan.py --path linlin64-128 --data centered_64-128 -genonly --width 64 -single -lindisc -lingen
