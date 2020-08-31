#!/bin/sh

EPOCHS=50

python flownet.py pyramid-END -train -pyramid -test --epochs $EPOCHS --train_mode END
python flownet.py pyramid-CONS -train -pyramid -test --epochs $EPOCHS --train_mode CONS
python flownet.py pyramid-COMB -train -pyramid -test --epochs $EPOCHS --train_mode COMB
python flownet.py pyramid-MIX -train -pyramid -test --epochs $EPOCHS --train_mode MIX
python flownet.py pyramid-GT -train -pyramid -test --epochs $EPOCHS --train_mode GT

python flownet.py linear-END -train -linear -test --epochs $EPOCHS --train_mode END
python flownet.py linear-CONS -train -linear -test --epochs $EPOCHS --train_mode CONS
python flownet.py linear-COMB -train -linear -test --epochs $EPOCHS --train_mode COMB
python flownet.py linear-MIX -train -linear -test --epochs $EPOCHS --train_mode MIX
python flownet.py linear-GT -train -linear -test --epochs $EPOCHS --train_mode GT

#python flownet.py sequ-END -train -sequ -test --epochs $EPOCHS --train_mode END
#python flownet.py sequ-CONS -train -sequ -test --epochs $EPOCHS --train_mode CONS
#python flownet.py sequ-COMB -train -sequ -test --epochs $EPOCHS --train_mode COMB
#python flownet.py sequ-MIX -train -sequ -test --epochs $EPOCHS --train_mode MIX
#python flownet.py sequ-GT -train -sequ -test --epochs $EPOCHS --train_mode GT

#python flownet.py spacial-END -train -test --epochs $EPOCHS --train_mode END
#python flownet.py spacial-CONS -train -test --epochs $EPOCHS --train_mode CONS
#python flownet.py spacial-COMB -train -test --epochs $EPOCHS --train_mode COMB
#python flownet.py spacial-MIX -train -test --epochs $EPOCHS --train_mode MIX
#python flownet.py spacial-GT -train -test --epochs $EPOCHS --train_mode GT