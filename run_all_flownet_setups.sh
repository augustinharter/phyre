#!/bin/sh

#conda init bash
#conda activate phyre
for i in 0 1 2 3 4 5 6 7 8 9
do
    if [ $1 = 'bigbase' ]
    then
        python solver.py --path VS-bigbase2 -train --folds 1 --foldstart $i -inspect --nper 32 --epochs 20 --lr 1e-3 --type withbase --train-mode WITHBASE -puretrain

    fi

    if [ $1 = 'deepex' ]
    then
        python solver.py --path VS-deepex -train --folds 1 --foldstart $i -inspect --nper 1 --epochs 20 --lr 1e-3 --type withbase --train-mode WITHBASE -puretrain -deepex

    fi

    if [ $1 = 'rads' ]
    then
        for i in 0 1 2 3 4 5 6 7 8 9
        do
            python solver.py --path VS-bigbase2 -load --from $i --run uni-epo$i --folds 1 --nper 32 --epochs 20 --lr 2e-3 --type withbase --train-mode WITHBASE -solve --radmode random -uniform
            python solver.py --path VS-bigbase2 -load --from $i --run cloud-epo$i --folds 1 --nper 32 --epochs 20 --lr 2e-3 --type withbase --train-mode WITHBASE -solve --radmode random
            python solver.py --path VS-bigbase2 -load --from $i --run grow-epo$i --folds 1 --nper 32 --epochs 20 --lr 2e-3 --type withbase --train-mode WITHBASE -solve --radmode old
        done



    fi

    if [ $1 = 'rad1' ]
    then
        echo "rad one"
        python solver.py --path VS-s-uni-mean -train -save -inspect --radmode mean -uniformform
        python solver.py --path VS-s-uni-median -train -save -inspect --radmode median -uniformform
        python solver.py --path VS-s-uni-rand -train -save -inspect --radmode random -formform
    fi

    if [ $1 = 'rad2' ]
    then
        echo "rad two"
        python solver.py --path VS-s-rad-mean -train -save -inspect --radmode mean
        python solver.py --path VS-s-rad-median -train -save -inspect --radmode median
        python solver.py --path VS-s-rad-rand -train -save -inspect --radmode random
    fi

    if [ $1 = 'one' ]
    then
        #python solver.py --path VS-withbase -train --folds 1 --foldstart $i -save -inspect --train-mode WITHBASE --type withbase
        #python solver.py --path VS-dijkstra -train --folds 1 --foldstart $i -save -inspect -dijkstra
        #python solver.py --path VS-templ-cons -train --folds 1 --foldstart $i -save -inspect --nper 32 -noshuff -pertempl
        #python solver.py --path VS-task-cons -train --folds 1 --foldstart $i -save -inspect --nper 32 -noshuff
        python solver.py --path VS-nper32 -train --folds 1 --foldstart $i -save -inspect --nper 32
        python solver.py --path VS-nobase -train --folds 1 --foldstart $i -save -inspect --train-mode NOBASE --type nobase
        python solver.py --path VS-direct -train --folds 1 --foldstart $i -save -inspect --train-mode DIRECT --type direct
    fi

    if [ $1 = 'two' ]
    then
        #python solver.py --path VS-dropout -train --folds 1 --foldstart $i -save -inspect -dropout
        python solver.py --path VS-lr01 -train --folds 1 --foldstart $i -save -inspect --lr 3e-4
        python solver.py --path VS-lr03 -train --folds 1 --foldstart $i -save -inspect --lr 1e-3
        python solver.py --path VS-lr3 -train --folds 1 --foldstart $i -save -inspect --lr 3e-3
        python solver.py --path VS-bs16 -train --folds 1 --foldstart $i -save -inspect --batchsize 16
        python solver.py --path VS-bs64 -train --folds 1 --foldstart $i -save -inspect --batchsize 64
        python solver.py --path VS-sched -train --folds 1 --foldstart $i -save -inspect -scheduled
    fi

    if [ $1 = 'three' ]
    then
        #python solver.py --path VS-x2 -train --folds 1 --foldstart $i -save -inspect --hidfac 2
        #python solver.py --path VS-x4 -train --folds 1 --foldstart $i -save -inspect --hidfac 4
        #python solver.py --path VS-x05 -train --folds 1 --foldstart $i -save -inspect --hidfac 0.5
        #python solver.py --path VS-neck -train --folds 1 --foldstart $i -save -inspect --neckfak 2
        #python solver.py --path VS-w128 -train --folds 1 --foldstart $i -save -inspect --width 128
        python solver.py --path VS-base -train --folds 1 --foldstart $i -save -inspect
        python solver.py --path VS-altconv -train --folds 1 --foldstart $i -save -inspect -altconv
    fi
done
