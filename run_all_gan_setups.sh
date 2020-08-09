#!/bin/sh

#conda init bash
#conda activate phyre

python action_cgan.py --path zoomed-single-linlin -single -lindisc -lingen --verbose 10 --saveevery 10 --epochs 50 --geneval 10
python action_cgan.py --path zoomed-double-linlin -lindisc -lingen --verbose 10 --saveevery 10 --epochs 50 --geneval 10
python action_cgan.py --path zoomed-single -single --verbose 10 --saveevery 10 --epochs 50 --geneval 10
python action_cgan.py --path zoomed-double --verbose 10 --saveevery 10 --epochs 50 --geneval 10