# Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL 



## Abstract

Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. 
In this paper, we propose a novel attacking algorithm which has an RL-based "director" searching for the optimal policy perturbation, and an "actor" crafting state perturbations following the directions from the director (i.e. the actor executes targeted attacks). Our proposed algorithm, PA-AD, is theoretically optimal against an RL agent and significantly improves the efficiency compared with prior RL-based works in environments with large or pixel state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in a wide range of environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



The implementation of A2C and ACKTR algorithms are based on an open source repository by Kostrikov, 2018.


## 1. Requirements/Installation

Please run the following command to install required packages (suggested python version: 3.7.0)

```
# requirements
pip install -r requirements.txt

# our packages
pip install -e .
```


## 2. Evaluating Pre-trained Models

**Download Our Pretrained Models**: We provide pre-trained models in folder *released_models* that can be downloaded [here](https://umd.box.com/s/61g7b3dwa5w0utap0056z6ake0fka925), where "dqn_victim" and "dqn_attacker" include victim and attacker models for our DQN experiments (Table 1), "a2c_victim" and "a2c_attacker" contain victim and attacker models for our A2C experiments (Table 1).

Due to the size limit of the supplementary material, the current *released_models* folder only contains pre-trained models in some environments, so that one can quickly check the performance of the attackers using the commands below. 
All pre-trained models will be released to the public after the paper's acceptance. If the readers would like to reproduce all the experimental results in our paper, please follow the 3rd part of this readme file to train the PA-AD attackers.


### DQN

- Use the following commands to evaluate our pre-trained PA-AD attacker against the DQN victim agent in PongNoFrameskip-v4:

```
cd paad_rl
python evaluator/test_attack.py --env-name PongNoFrameskip-v4 --v-type dqn --v-path released_models/dqn_victim/PongNoFrameskip-v4 --attacker paad --det --attack-model released_models/dqn_attacker/dqn_obspol_attacker_PongNoFrameskip-v4_e0.0002_fgsm --epsilon 0.0002 --test-episodes 100
```
This will rollout the PA-AD attacker that has been trained already and stored in *released_models/dqn_attacker/*. Here, we rollout the attacker for 100 episodes. But for the results reported in our experiment section, we use 1000 test episodes to further make sure that randomness does not affect our comparison. Feel free to change the number after --test-episodes in case you want to rollout more episodes.

- Use the following commands to evaluate our pre-trained PA-AD attacker against the DQN victim agent in BoxingNoFrameskip-v4:

```
cd paad_rl
python evaluator/test_attack.py --env-name BoxingNoFrameskip-v4 --v-type dqn --v-path released_models/dqn_victim/BoxingNoFrameskip-v4 --attacker paad --det --attack-model released_models/dqn_attacker/dqn_obspol_attacker_BoxingNoFrameskip-v4_e0.001_fgsm --epsilon 0.001 --test-episodes 100
```

- Use the following commands to evaluate our pre-trained PA-AD attacker against the DQN victim agent in TutankhamNoFrameskip-v4:

```
cd paad_rl
python evaluator/test_attack.py --env-name TutankhamNoFrameskip-v4 --v-type dqn --v-path released_models/dqn_victim/TutankhamNoFrameskip-v4 --attacker paad --det --attack-model released_models/dqn_attacker/dqn_obspol_attacker_TutankhamNoFrameskip-v4_e0.00075_fgsm --epsilon 0.00075 --test-episodes 100
```


- To test heuristic attack methods in PongNoFrameskip-v4, use the following commands:

```
cd paad_rl
python evaluator/test_attack.py --env-name PongNoFrameskip-v4 --v-type dqn --v-path released_models/dqn_victim/PongNoFrameskip-v4 --attacker minbest --epsilon 0.0002  --test-episodes 100
```
(the option --attacker can be one of "minbest", "momentum", "minq", "maxdiff", "random")

- To test heuristic attack methods in BoxingNoFrameskip-v4, use the following commands:

```
cd paad_rl
python evaluator/test_attack.py --env-name BoxingNoFrameskip-v4 --v-type dqn --v-path released_models/dqn_victim/BoxingNoFrameskip-v4 --attacker minbest --epsilon 0.001 --test-episodes 100
```

- To test heuristic attack methods in TutankhamNoFrameskip-v4, use the following commands:

```
cd paad_rl
python evaluator/test_attack.py --env-name TutankhamNoFrameskip-v4 --v-type dqn --v-path released_models/dqn_victim/TutankhamNoFrameskip-v4 --attacker minbest --epsilon 0.00075 --test-episodes 100
```

To use environments that we haven't provided in our released models or test different values of epsilon, please see section 3.

### A2C

- Use the following commands to evaluate our pre-trained PA-AD attacker against the A2C victim agent in AlienNoFrameskip-v4:

```
cd paad_rl
python evaluator/test_attack.py --env-name AlienNoFrameskip-v4 --v-path released_models/a2c_victim/AlienNoFrameskip-v4 --attacker paad --det --attack-model released_models/a2c_attacker/obspol_attacker_AlienNoFrameskip-v4_e0.001_fgsm --epsilon 0.001 --test-episodes 100
```

- To test heuristic attack methods, use the following commands:

```
cd paad_rl
python evaluator/test_attack.py --env-name AlienNoFrameskip-v4 --v-path released_models/a2c_victim/AlienNoFrameskip-v4 --attacker minbest --epsilon 0.001 --test-episodes 100
```

(the option --attacker can be one of "minbest", "momentum", "maxdiff", "random")

- We also provide pre-trained models in SeaquestNoFrameskip-v4 and TutankhamNoFrameskip-v4:
```
# PA-AD Attacker in Seaquest:
cd paad_rl
python evaluator/test_attack.py --env-name SeaquestNoFrameskip-v4 --v-path released_models/a2c_victim/SeaquestNoFrameskip-v4 --attacker paad --det --attack-model released_models/a2c_attacker/obspol_attacker_SeaquestNoFrameskip-v4_e0.005_fgsm --epsilon 0.005 --test-episodes 100
```

```
# PA-AD Attacker in Tutankham:
cd paad_rl
python evaluator/test_attack.py --env-name TutankhamNoFrameskip-v4 --v-path released_models/a2c_victim/TutankhamNoFrameskip-v4 --attacker paad --det --attack-model released_models/a2c_attacker/obspol_attacker_TutankhamNoFrameskip-v4_e0.001_fgsm --epsilon 0.001 --test-episodes 100
```


## 3. Training PA-AD Attackers and Reproducing All Attacker Results

### DQN


- Train and evaluate all attackers in Pong (to reproduce the row of DQN-Pong in our Table 1):

```
cd paad_rl
bash scripts/train/dqn_train_attacker.sh PongNoFrameskip-v4 0
```
Again here 0 is the cuda id, and it will also create a folder at *data/dqn_results/pong* if it doesn't exist already. We use ACKTR to train the director or PA-AD for 6 million frames. However, most likely it will take much shorter for our PA-AD algorithm to converge. So to save time, feel free to try smaller number of frames by changing the values of STEPS in *dqn_train_attacker.sh*. 

(Note: If there is no GPU, please replace --cuda-id ${CUDA} with --no-cuda in each command in the script files.)

- Train and evaluate paad attacker for Boxing:

```
cd paad_rl
bash scripts/train/dqn_train_attacker.sh BoxingNoFrameskip-v4 0
```

- Train and evaluate paad attacker for Tutankham:

```
cd paad_rl
bash scripts/train/dqn_train_attacker.sh TutankhamNoFrameskip-v4 0
```

- Train and attack a new DQN Victim (taking Pong as an example):

If you would like to train a new DQN victim from scratch and test how it performs under various attackers, please run

```
cd paad_rl
bash scripts/train/dqn_train_victim.sh PongNoFrameskip-v4 0
```

Here 0 is the cuda id. The training script will save the model at *data/learned_models/dqn/*.<br>
Note that DQN training usually takes about 3-4 days to finish, which is much longer than the A2C training. In our experiment, we have trained our DQN agent for 6 million frames. But feel free to change the argument frame_total in the above script in case you want the victim agent to be trained for more or less frames. The hyperparameters used in the DQN victim agent training follow the exact same from the original deepmind papers. For the detailed explanation of these adjustable parameters, please read *trainer_victim/dqn.py*

Then, to attack a victim model you just trained, please comment out Line 9 and uncomment Line 10 in *scripts/train/dqn_train_attacker.sh* and run it.


### A2C


- Train and evaluate all attackers in Alien (to reproduce the row of A2C-Alien in our Table 1):

```
cd paad_rl
bash scripts/train/a2c_train_attacker.sh AlienNoFrameskip-v4 0
```
where 0 is the cuda id. 

This command will automatically run all the attackers against the victim saved in *released_models/a2c_victim*. 
The trained attacker model will be saved at *learned_adv/acktr/*. The episode rewards of the victim agent under different attackers will be saved in folder *data/a2c_results/alien*. See *scripts/train/a2c_train_attacker.sh* for more details.


- Train and attack a new A2C victim:

If you would like to train a new A2C victim and test how it performs under various attackers, we also provide the following command:
```
cd paad_rl
bash scripts/train/a2c_train_victim.sh BreakoutNoFrameskip-v4 0
```
where 0 is the cuda id. The trained model will be saved in folder *learned_models/a2c*.

Then, to attack a victim model you just trained, please comment out Line 9 and uncomment Line 10 in *scripts/train/a2c_train_attacker.sh* and run 
```
bash scripts/train/dqn_train_attacker.sh BreakoutNoFrameskip-v4 0
```


## 4. Attacking Robust Models

This section is corresponding to our Table 5 in Appendix F.2.4., where we show the performance of our PA-AD attacker agains SA-DQN and RADIAL-RL robust models in Atari games. All victim models are provided by the authors of SA-DQN and RADIAL-RL. Due to the size limit of the supplementary material, we include some of the the pre-trained victims and attackers. 

### (1) SA-DQN (Zhang et al. 2020)

A pre-trained SA-DQN model in RoadRunner is saved at *released_models/sa_models*, which is provided by Zhang et al. 
Our corresponding PA-AD attacker is in *released_models/robust_attacker/sa*.

**Note**: The original implementation of SA-DQN (Zhang et al. 2020) requires the [auto-LiRPA](https://github.com/KaidiXu/auto_LiRPA) package (Kaidi Xu, et al. 2020), which is an open-source codebase and can be installed with the following commands.

```
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install
```

**Test Pre-trained Attackers**:

The following command will evaluate all the attackers against the pre-trained SA-DQN model in RoadRunner.
```
cd paad_rl
bash scripts/attack_robust_models/evaluate_attack_sa.sh RoadRunnerNoFrameskip-v4 0
```
where 0 is the cuda id. The attacker results will be saved in folder *data/robust_results/*.

**Train Attackers**:

To train our PA-AD attacker with $\epsilon=1/255$, please use the following script: 

```
cd paad_rl
bash scripts/attack_robust_models/train_attack_sa.sh RoadRunnerNoFrameskip-v4 0
```

The trained attacker model will be saved at *learned_adv/ppo/*.


### (2) RADIAL-RL (Oikarinen et al. 2020)

A pre-trained RADIAL-DQN model in RoadRunner is saved at *released_models/radial_models/dqn*, which is provided by Oikarinen et al. Our corresponding PA-AD attackers are in *released_models/robust_attacker/radial* (attackers for $\epsilon=1/255$ and $\epsilon=3/255$, respectively).

**Test Pre-trained Attackers**:

The following command will evaluate all the attackers against the pre-trained RADIAL-DQN model in RoadRunner with budget $\epsilon=1/255$ and $\epsilon=3/255$.
```
cd paad_rl
bash scripts/attack_robust_models/evaluate_attack_radial.sh RoadRunnerNoFrameskip-v4 0 dqn
```

where 0 is the cuda id. The attacker results will be saved in folder *data/robust_results/*.

**Train Attackers**:

One can use the following command to train the PA-AD attacker.
```
cd paad_rl
bash scripts/attack_robust_models/train_attack_radial.sh RoadRunnerNoFrameskip-v4 0 dqn 0.00392
```

(Note: 0.00392=1/255, 0.01176=3/255)

The trained attacker model will be saved at *learned_adv/ppo/*.


## 5. Alternating Training 

- Train a robust A2C model using ATLA (Zhang et al. 2021) via the following commands in BankHeist:

```
cd paad_rl
python trainer_atla/a2c_atla_pa.py --env-name 'BankHeistNoFrameskip-v4' --cuda-id 0 --epsilon 0.01176 --num-process 16 --log-interval 2 --save-interval 20 --fgsm --num-steps 64 --use-linear-lr-decay --res-dir './data/a2c_results/'
```
The *data/a2c_results/* document only records the training data of ATLA. The trained ATLA policy models will be saved in *learned_models/a2c/*.


**References:**

Ilya Kostrikov. Pytorch implementations of reinforcement learning algorithms. https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Duane Boning, and Cho-Jui Hsieh. Robust deep reinforcement learning against adversarial perturbations on observations. arXiv preprint arXiv:2003.08938, 2020.

Tuomas Oikarinen, Tsui-Wei Weng, and Luca Daniel. Robust deep reinforcement learning through adversarial loss. arXiv preprint arXiv:2008.01976, 2020.

Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh. Automatic perturbation analysis for scalable certified robustness and beyond. Advances in Neural Information Processing Systems, 33, 2020.

Huan Zhang, Hongge Chen, Duane Boning, Cho-Jui Hsieh. Robust reinforcement learning on state observations with learned optimal adversary. arXiv preprint arXiv:2101.08452, 2021.
