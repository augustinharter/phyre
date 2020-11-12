#!/bin/sh

#conda init bash
#conda activate phyre
if [ $1 = 'data' ]
then
    python solver.py --path VS-bigbase2 --folds 1 -train --foldstart 0 --nper 32 --epochs 0 --lr 1e-3 --type withbase --train-mode WITHBASE -puretrain
    python solver.py --path VS-bigbase2 --folds 1 -train --foldstart 0 --nper 100 --epochs 0 --lr 1e-3 --type withbase --train-mode WITHBASE -puretrain

fi

if [ $1 = 'percent' ]
then
    #python solver.py --path GEN-BASE --folds 1 -train -inspect --foldstart $i --nper 10 --epochs 10 -puretrain --run percent5
    python solver.py --path GEN-BASE --folds 1 -load -solve --from 9 --foldstart $i --nper 10 --epochs 10 --run percent5

fi


if [ $1 = 'comb-train' ]
then
    for i in 1 2 3 4 5 6 7 8 9
    do
        python solver.py --path GEN-COMB --folds 1 -train -inspect --foldstart $i --nper 10 --epochs 30 -puretrain --run train --train-mode COMB --epochstart 20
    done

fi

if [ $1 = 'comb-inspect' ]
then
    for i in 1
    do
        python solver.py --path GEN-COMB --folds 1 -load --from 10 -inspect --foldstart $i --nper 10 --epochs 30 --run inspect --train-mode COMB --epochstart 0 -singleviz
    done

fi

if [ $1 = 'comb-solve1' ]
then
    for j in 1 2 3 5 7 9 12 15 19
    do
        for i in 0 1 2 3 4 5 6 7 8 9
        do
            python solver.py --path GEN-COMB --folds 1 -load --from $j -solve-templ --foldstart $i --nper 10 --run solve --train-mode COMB
        done
    done

fi


if [ $1 = 'comb-solve2' ]
then
    for j in 4 6 8 10 11 12 13 14 16 17 18
    do
        for i in 0 1 2 3 4 5 6 7 8 9
        do
            python solver.py --path GEN-COMB --folds 1 -load --from $j -solve-templ --foldstart $i --nper 10 --run solve --train-mode COMB
        done
    done
fi

if [ $1 = 'comb-solve3' ]
then
    for j in 20 21 22 23 24 25 26 27 28 29
    do
        for i in 0 1 2 3 4 5 6 7 8 9
        do
            python solver.py --path GEN-COMB --folds 1 -load --from $j -solve-templ --foldstart $i --nper 10 --run solve --train-mode COMB
        done
    done
fi


if [ $1 = 'deepex' ]
then
    python solver.py --path VS-deepex -train --folds 1 --foldstart $i -inspect --nper 10 --epochs 10 --type withbase --train-mode WITHBASE -puretrain
    python solver.py --path VS-deepex -load --from 4 --folds 1 --foldstart $i --nper 10 --epochs 25 --lr 3e-3 --type withbase --train-mode WITHBASE -dtrain -deepex

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
    #python solver.py --path VS-altconv -train --folds 1 --foldstart $i -save -inspect -altconv
fi
