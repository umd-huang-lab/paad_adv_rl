#! /bin/bash  

ENV=$1
CUDA=$2
VICTIM=$3
EPS=$4

python attack_robust/radial_train_attacker.py --env-name ${ENV} --cuda-id ${CUDA} --epsilon ${EPS} --v-algo ${VICTIM} --algo ppo --num-steps 64 --lr 5e-4 --use-linear-lr-decay --clip-param 0.1 --num-env-steps 1000000 --attack-steps 30 