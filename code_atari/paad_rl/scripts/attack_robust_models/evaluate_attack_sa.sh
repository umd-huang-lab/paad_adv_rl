#! /bin/bash  

ENV=$1
CUDA=$2
EPISODES=50
EPS=0.00392

DIR="./data/robust_results/"

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir ${DIR}
fi


for ATTACKER in paad minbest momentum minq maxdiff random
do
    python attack_robust/sa_evaluate_attack.py --env-name ${ENV} --cuda-id ${CUDA} --test-episodes ${EPISODES} \
    --epsilon ${EPS} --attacker ${ATTACKER} --attack-steps 30 --log-interval 1000 --cuda-deterministic  \
    > ${DIR}/${ENV}_sa_${ATTACKER}_e${EPS}.txt 
done