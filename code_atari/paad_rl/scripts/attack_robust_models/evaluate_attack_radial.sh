#! /bin/bash  

ENV=$1
CUDA=$2
VICTIM=$3
EPISODES=50

DIR="./data/robust_results/"

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    echo "create a new dir"
    mkdir ${DIR}
fi

for EPS in 0.00392 0.01176
do
    for ATTACKER in paad minbest momentum minq maxdiff random
    do
        python attack_robust/radial_evaluate_attack.py --env-name ${ENV} --cuda-id ${CUDA} --test-episodes ${EPISODES} \
        --epsilon ${EPS} --attacker ${ATTACKER} --attack-steps 30 --v-algo ${VICTIM} --log-interval 1000 --cuda-deterministic  \
        > ${DIR}/${ENV}_radial_${VICTIM}_${ATTACKER}_e${EPS}.txt 
    done
done