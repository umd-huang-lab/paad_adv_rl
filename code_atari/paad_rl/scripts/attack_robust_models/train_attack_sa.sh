#! /bin/bash  

ENV=$1
CUDA=$2
EPS=0.00392

python attack_robust/sa_train_attacker.py --env-name ${ENV} --cuda-id ${CUDA} --epsilon ${EPS} --algo ppo --num-steps 64 --lr 5e-4 --use-linear-lr-decay --clip-param 0.1 --num-env-steps 1000000 --attack-steps 30 