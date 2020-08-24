#!/bin/sh

python flownet.py spacial-GT-trans-10 -train -GT -test --epochs 10
#python flownet.py sequ-GT-10 -train -GT -sequ -test --epochs 10