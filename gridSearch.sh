#!/bin/bash

LRS=( .5 1 2 )
MUS=( 0 )
RHOS=( 0 )
OPTS=( Anneal )
for OPT in "${OPTS[@]}"
do
  for RHO in "${RHOS[@]}"
  do
    for MU in "${MUS[@]}"
    do
      for LR in "${LRS[@]}"
      do
        python glove.py -o $OPT -eta $LR -rho $RHO -n 12 -mu $MU
      done
    done
  done
done

#LRS=( .005 )
#MUS=( .25 )
#RHOS=( .5 )
#OPTS=( Adam NAdam AdaMax NAdaMax )
#for OPT in "${OPTS[@]}"
#do
#  for RHO in "${RHOS[@]}"
#  do
#    for MU in "${MUS[@]}"
#    do
#      for LR in "${LRS[@]}"
#      do
#        python glove.py -o $OPT -eta $LR -rho $RHO -n 12 -mu $MU
#      done
#    done
#  done
#done
#
#LRS=( 1.0 )
#MUS=( .5 )
#RHOS=( 0 )
#OPTS=( Momentum NAG )
#for OPT in "${OPTS[@]}"
#do
#  for RHO in "${RHOS[@]}"
#  do
#    for MU in "${MUS[@]}"
#    do
#      for LR in "${LRS[@]}"
#      do
#        python glove.py -o $OPT -eta $LR -rho $RHO -n 12 -mu $MU
#      done
#    done
#  done
#done
#
#LRS=( .002 .005 )
#MUS=( 0 )
#RHOS=( .67 )
#OPTS=( RMSProp MMAProp )
#for OPT in "${OPTS[@]}"
#do
#  for RHO in "${RHOS[@]}"
#  do
#    for MU in "${MUS[@]}"
#    do
#      for LR in "${LRS[@]}"
#      do
#        python glove.py -o $OPT -eta $LR -rho $RHO -n 12 -mu $MU
#      done
#    done
#  done
#done

#python glove.py -o SGD      -eta .2 -n 12
#python glove.py -o Anneal   -eta .2 -n 12
#python glove.py -o Momentum -eta 1.0  -mu .5   -n 12
#python glove.py -o NAG      -eta 1.0  -mu .5   -n 12
#python glove.py -o RMSProp  -eta .002 -rho .67 -n 12
#python glove.py -o MMAProp  -eta .005 -rho .67 -n 12
#python glove.py -o Adam     -eta .005 -mu .25  -rho .5 -n 12
#python glove.py -o NAdam    -eta .005 -mu .25  -rho .5 -n 12
#python glove.py -o AdaMax   -eta .005 -mu .25  -rho .5 -n 12
#python glove.py -o NAdaMax  -eta .005 -mu .25  -rho .5 -n 12