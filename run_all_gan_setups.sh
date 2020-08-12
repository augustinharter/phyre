#!/bin/sh

#conda init bash
#conda activate phyre

python action_cgan.py --path full-single-static --data scene --width 64 --epochs 150 -single
python action_cgan.py --path full-single-static --data scene -genonly --width 64 --epochs 150 -single

python action_cgan.py --path 32-single-static --data centered_32x32 --width 32 --epochs 150 -single
python action_cgan.py --path 32-single-static --data scene -genonly --width 32 -single

python action_cgan.py --path 32-single-dyn --data centered_32x32_dyn --width 32 --epochs 150 -single
python action_cgan.py --path 32-single-dyn --data scene -genonly --width 32 -single