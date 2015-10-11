#!/bin/bash

#models=( FastGRU )
#funcs=( resqrt )
#opts=( RMSProp AdaDelta Adam )
#
#for model in "${models[@]}"
#do
#  for func in "${funcs[@]}"
#  do
#    for opt in "${opts[@]}"
#    do
#      python UNKRNN.py -m $model -f $func -o $opt -p "/home/tdozat/Scratch/UNK/data/" -s 200
#    done
#  done
#done
#
#LRS=( 1 2 5 )
#OPTS=( NAG )
#for LR in "${LRS[@]}"
#do
#  for OPT in "${OPTS[@]}"
#  do
#    python glove.py -o $OPT -l $LR
#  done
#done

#LRS=( 1e-3 2e-3 1e-2 )
#OPTS=( RMSProp Adam AdaMax )
#for LR in "${LRS[@]}"
#do
#  for OPT in "${OPTS[@]}"
#  do
#    python glove.py -o $OPT -l $LR
#  done
#done

#python glove.py -o AdaMax -l .005 -e 50 -m spd
#python glove.py -o AdaMax -l .005 -e 50 -m sd
#python glove.py -o AdaMax -l .005 -e 50 -m sp
#python glove.py -o AdaMax -l .002 -e 100 -m s

#EPSILONS=( 1e-4 1e-3 )
#LRS=( 5e-4 1e-3 2e-3 5e-3 )
#
#for EPSILON in "${EPSILONS[@]}"
#do
#  for LR in "${LRS[@]}"
#  do
#    python glove.py -l $LR -e $EPSILON
#  done
#done

#python glove.py -n 50 -m spd
#python glove.py -n 50 -m sd
python glove.py -n 50 -m sp
python glove.py -n 100 -m s