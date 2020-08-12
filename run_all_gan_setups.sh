#!/bin/sh

#conda init bash
#conda activate phyre

#python action_cgan.py --path zoomed-single-linlin -single -lindisc -lingen --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path zoomed-single-linlin -genonly -single -lindisc -lingen --verbose 10 --saveevery 10 --geneval 10

#python action_cgan.py --path zoomed-double-linlin -lindisc -lingen --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path zoomed-double-linlin -genonly -lindisc -lingen --verbose 10 --saveevery 10 --epochs 100 --geneval 10

#python action_cgan.py --path zoomed-single -single --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path zoomed-single -genonly -single --verbose 10 --saveevery 10 --geneval 10

#python action_cgan.py --path zoomed-double --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path zoomed-double -genonly --verbose 10 --saveevery 10 --epochs 100 --geneval 10

#python action_cgan.py --path zoomed-staged -sequ --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path zoomed-staged -genonly -sequ --verbose 1 --saveevery 10 --epochs 150 --geneval 10

#python action_cgan.py --path full-double-linlin -lingen -lindisc -full --width 64 --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path full-double-linlin -lingen -lindisc -full -genonly --width 64

python action_cgan.py --path full-double-higher-res-encoder -full --width 64 --verbose 10 --saveevery 10 --epochs 100 --geneval 10
python action_cgan.py --path full-double-higher-res-encoder -full -genonly --width 64

#python action_cgan.py --path full-staged -full -sequ --width 64 --verbose 10 --saveevery 10 --epochs 100 --geneval 10
#python action_cgan.py --path full-staged -full -genonly -sequ --width 64