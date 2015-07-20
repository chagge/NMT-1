#!/bin/bash

models=( FastGRU )
funcs=( resqrt )
opts=( RMSProp AdaDelta Adam )

for model in "${models[@]}"
do
  for func in "${funcs[@]}"
  do
    for opt in "${opts[@]}"
    do
      python UNKRNN.py -m $model -f $func -o $opt -p "/home/tdozat/Scratch/UNK/data/" -s 200
    done
  done
done
