#!/bin/sh

EPOCHS=1

python flownet.py sequ-END -train -sequ -test --epochs $EPOCHS --train_mode END
python flownet.py sequ-CONS -train -sequ -test --epochs $EPOCHS --train_mode CONS
python flownet.py sequ-COMB -train -sequ -test --epochs $EPOCHS --train_mode COMB
python flownet.py sequ-MIX -train -sequ -test --epochs $EPOCHS --train_mode MIX
python flownet.py sequ-GT -train -sequ -test --epochs $EPOCHS --train_mode GT

python flownet.py spacial-END -train -test --epochs $EPOCHS --train_mode END
python flownet.py spacial-CONS -train -test --epochs $EPOCHS --train_mode CONS
python flownet.py spacial-COMB -train -test --epochs $EPOCHS --train_mode COMB
python flownet.py spacial-MIX -train -test --epochs $EPOCHS --train_mode MIX
python flownet.py spacial-GT -train -test --epochs $EPOCHS --train_mode GT