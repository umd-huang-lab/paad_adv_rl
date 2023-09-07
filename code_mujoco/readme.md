# Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL 



## Abstract

Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. 
In this paper, we propose a novel attacking algorithm which has an RL-based "director" searching for the optimal policy perturbation, and an "actor" crafting state perturbations following the directions from the director (i.e. the actor executes targeted attacks). Our proposed algorithm, PA-AD, is theoretically optimal against an RL agent and significantly improves the efficiency compared with prior RL-based works in environments with large or pixel state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in a wide range of environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



This document contains a reference implementation for alternating training of learned adversaries (ATLA) with our PA-AD attack method. Our code is based on ATLA (Zhang et al. 2021) codebase: [ATLA](https://github.com/huanzhang12/ATLA_robust_RL)


## 1. Requirements/Installation

Please run the following command to install required packages (suggested python version: 3.7.0)

```
# requirements
pip install -r requirements.txt

# need auto_LiRPA (Kaidi Xu, et al. 2020)
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install

# enter the trainer document for ATLA
cd ../trainer_atla
```

## 2. Training PA-AD Attackers for Pretrained Models

**Downloading Our Pretrained Models**: All of the trained models can be downloaded [here](https://umd.box.com/s/lezv2f74w3zvgz7n791j0xlkiz2enybw). Please download and place the *models* folder under the root folder *trainer_atla*.

The commands provide the best hyperparameters for the adversary training via a hyperparameter grid search.

### PPO
- Use the following commands to train PA-AD attacker against the vanilla PPO agent in Ant:

```
python run.py --config-path configs/config_ant_vanilla_ppo.json --load-model models/PPO/model-ppo-ant.model --mode adv_pa_ppo --ppo-lr-adam 0.0 --adv-ppo-lr-adam 3e-5 --adv-val-lr 3e-5 --adv-entropy-coeff 0.0 --adv-clip-eps 0.4
```

This will save an experiment folder at vanilla_ppo_ant/agents/YOUR_EXP_ID, where YOUR_EXP_ID is a randomly generated experiment ID. You can extract the best model from this folder by running:

```
python get_best_pickle.py vanilla_ppo_ant/agents/YOUR_EXP_ID
```
which will generate an adversary model best_model.YOUR_EXP_ID.model, for example best_model.7d48fb45.model.

The trained best attacker for vanilla PPO has been stored in *models/ppo_attacker/*.
To train for different MuJoCo environment, simply change the config_path and load_model in the command above and switch epsilon to the value that we report in our paper. 

- Evaluating the trained attacker:

```
python test.py --config-path configs/config_ant_vanilla_ppo.json --load-model models/PPO/model-ppo-ant.model --deterministic --attack-method paadvpolicy --attack-advpolicy-network models/ppo_attacker/best_attacker_ant.model
```

- Other pretrained attackers can be referred to ATLA repository.

## 3. Training Models with Alternating Training

### ATLA
- Alternating training with PA-AD attack mode in Ant:

```
python run.py --config-path configs/config_ant_pa_atla_ppo.json
```

Change ant to other MuJoCo environments names to run other environments.

Training results will be saved to a directory specified by the out_dir parameter in the json file.
To allow multiple runs, each experiment is assigned a unique experiment ID, which is saved as a folder under out_dir (e.g., pa_atla_ppo_ant/agents/YOUR_EXP_ID).

- Testing the ATLA models under no attack:

```
# Change the --exp-id to match the folder name in pa_atla_ppo_ant/agents
python test.py --config-path configs/config_ant_pa_atla_ppo.json --exp-id YOUR_EXP_ID --deterministic
```

- Extracting the best ATLA model including the policy model and the adversary model from this folder by running: 

```
python get_best_pickle.py pa_atla_ppo_ant/agents/YOUR_EXP_ID
```
which will generate an ATLA model best_model.YOUR_EXP_ID.model.


## 4. Evaluating ATLA models under different attackers

- Testing the ATLA models with trained attacker:

```
python test.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --deterministic --attack-method paadvpolicy --attack-advpolicy-network models/ppo_attacker/best_attacker_ant.model
```

- You can also train attackers for the trained ATLA models.

Train **PA-AD attacker** for ATLA models:
```
python run.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --mode adv_pa_ppo --ppo-lr-adam 0.0 --adv-ppo-lr-adam 3e-5 --adv-val-lr 3e-5 --adv-entropy-coeff 0.0 --adv-clip-eps 0.4
```

Train **SA-RL (Zhang et al. 2020) attacker** for ATLA models:
```
python run.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --mode adv_ppo --ppo-lr-adam 0.0 --adv-ppo-lr-adam 3e-5 --adv-val-lr 3e-5 --adv-entropy-coeff 0.0 --adv-clip-eps 0.4
```

Train **Robust Sarsa (RS) (Zhang et al. 2020) Attacker** for ATLA models:
```
# Step 1:
python test.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --sarsa-enable --sarsa-model-path sarsa_ant_pa_atla_ppo.model
# Step 2:
python test.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --attack-eps=0.15 --attack-method sarsa --attack-sarsa-network sarsa_ant_pa_atla_ppo.model --deterministic
```

- Attacking ATLA models with other attackers

**Random attacker**
```
python test.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --attack-eps=0.15 --attack-method random --deterministic
```

**Maximal Action Difference (MAD) (Zhang et al. 2020) Attacker**
```
python test.py --config-path configs/config_ant_pa_atla_ppo.json --load-model best_model.YOUR_EXP_ID.model --attack-eps=0.15 --attack-method action --deterministic
```

**Pretrained Models**

We have updated several pretrained models for PA-ATLA-PPO and present their performance below. It's important to note that these pretrained models were selected randomly from training runs using the best hyperparameters. In RL algorithms, variance across training runs can be substantial. Therefore, to provide a robust evaluation, we conducted 30 training runs for each agent configuration. The reported performance metrics in our paper represent the median performance under the strongest attacks, rather than the best or worst case scenarios. Consequently, there may be difference but in variance between our reported results and the performance of the pretrained models.

| Environment        | No attack | Heuristic attack | Evasion attack |
| ------------------ | --------- | ---------------- | -------------- |
| Ant-v2 (pertained) | 5329      | 3920             | 3518           |
| Reported           | 5469      | 4124             | 2986           |
| HalfCheetah-v2     | 6185      | 5684             | 4472           |
| Reported           | 6289      | 5226             | 3840           |
| Hopper-v2          | 3594      | 2866             | 2293           |
| Reported           | 3449      | 3002             | 1529           |



**References:**

Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Duane Boning, and Cho-Jui Hsieh. Robust deep reinforcement learning against adversarial perturbations on observations. arXiv preprint arXiv:2003.08938, 2020.

Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang, Minlie Huang, Bhavya Kailkhura, Xue Lin, and Cho-Jui Hsieh. Automatic perturbation analysis for scalable certified robustness and beyond. Advances in Neural Information Processing Systems, 33, 2020.

Huan Zhang, Hongge Chen, Duane Boning, Cho-Jui Hsieh. Robust reinforcement learning on state observations with learned optimal adversary. arXiv preprint arXiv:2101.08452, 2021.

