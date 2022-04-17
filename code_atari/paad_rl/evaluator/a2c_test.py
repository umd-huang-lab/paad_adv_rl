import copy
import glob
import os
import sys
import time
from collections import deque

import gym
from gym.spaces.box import Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.arguments import get_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate
from paad_rl.attacker.attacker import *
from paad_rl.a2c_ppo_acktr.algo.kfac import KFACOptimizer


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    print("The observation space is", envs.observation_space)

    ## load victim
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    agent_states, ob_rms = torch.load(os.path.join(args.victim_dir, args.env_name),
            map_location=device)
    actor_critic.load_state_dict(agent_states)
    actor_critic.to(device)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        # vec_norm.ob_rms = ob_rms

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    if args.attacker:
        vtype = "_vdet" if args.v_det else ""
        optim_method = "momentum" if args.momentum else ("fgsm" if args.fgsm else "pgd" )
        rew_file = open(os.path.join(args.res_dir, r"{}_{}_{}_e{}_{}{}.txt".format(args.v_algo, 
                args.env_name, args.attacker, args.epsilon, optim_method, vtype)), "wt")
        if args.attacker == "minbest":
            attacker = Huang_Attack()
        elif args.attacker == "minq":
            attacker = Pattanaik_Attack()
        elif args.attacker == "maxdiff":
            attacker = KL_Attack()
        elif args.attacker == "maxworst":
            attacker = MaxWorst_Attack()
        elif args.attacker == "randworst":
            attacker = RandomWorst_Attack()
        elif args.attacker == "random":
            attacker = Random_Attack()
        elif args.attacker == "targetworst":
            print("load target worst attacker")
            exp_name = "learned_worst/a2c/{}".format(args.env_name)
            target_attacker = Policy(
                envs.observation_space.shape,
                envs.action_space,
                beta=False,
                base_kwargs={'recurrent': False})
            attacker_state, _ = torch.load(exp_name, map_location=device)
            target_attacker.load_state_dict(attacker_state)
            target_attacker.to(device)
            attacker = TargetWorst_Attack(target_attacker)
        elif args.attacker == "sarl":
            print("load obs attacker")
            exp_name = "obs_attacker_{}_e{}_{}{}".format(args.env_name, args.epsilon,
                "fgsm" if args.fgsm else "pgd", vtype)
            action_space = Box(-args.epsilon, args.epsilon, envs.observation_space.shape)
            obs_attacker = Policy(
                envs.observation_space.shape,
                action_space,
                beta=False,
                epsilon=args.epsilon,
                base_kwargs={'recurrent': False})
            old_steps, obs_attacker_state, _ = \
                    torch.load(os.path.join(args.adv_dir, args.algo, 
                        exp_name), map_location=device)
            obs_attacker.load_state_dict(obs_attacker_state)
            obs_attacker.to(device)
            attacker = Obs_Attack(obs_attacker, envs.observation_space.shape, det=args.det)
            print("training steps for this model:", old_steps)
        elif args.attacker == "paad":
            print("load observation-policy attacker")
            exp_name = "obspol_attacker_{}_e{}_{}{}".format(args.env_name, args.epsilon, optim_method, vtype)
            if envs.action_space.__class__.__name__ == "Discrete":
                action_space = Box(-1.0, 1.0, (envs.action_space.n-1,))
                cont = False
            elif envs.action_space.__class__.__name__ == "Box":
                action_space = Box(-1.0, 1.0, (envs.action_space.shape[0],))
                cont = True
            pa_attacker = Policy(
                envs.observation_space.shape,
                action_space,
                beta=False,
                epsilon=args.epsilon,
                base_kwargs={'recurrent': False})
            if args.algo == "acktr":
                KFACOptimizer(pa_attacker) # the model structure for the acktr attacker is different
            old_steps, pa_attacker_state, _ = \
                    torch.load(os.path.join(args.adv_dir, args.algo,
                        exp_name), map_location=device)
            pa_attacker.load_state_dict(pa_attacker_state)
            pa_attacker.to(device)
            attacker = ObsPol_Attack(pa_attacker, det=args.det, cont=cont)
            print("training steps for this model:", old_steps)
    
    else:
        rew_file = open(os.path.join(args.res_dir, r"{}_{}_noattack.txt".format(args.v_algo, args.env_name)), "wt")
    
    obs = envs.reset()

    ## Attack obs (if any)
    if args.attacker:
        if args.attacker == "sarl":
            obs = attacker.attack_stoc(obs, rollouts.recurrent_hidden_states[0],
                        rollouts.masks[0], epsilon=args.epsilon, device=device)
        elif args.attacker == "random":
            obs = attacker.attack_stoc(obs, epsilon=args.epsilon, device=device)
        else:
            obs = attacker.attack_stoc(actor_critic, obs, rollouts.recurrent_hidden_states[0],
                        rollouts.masks[0], epsilon=args.epsilon, fgsm=args.fgsm, lr=args.attack_lr,
                        pgd_steps=args.attack_steps, device=device, rand_init=args.rand_init, 
                        momentum=args.momentum)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    num_episodes = 0
    all_rewards = []
    
    for j in range(num_updates):
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], deterministic=args.v_det)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    num_episodes += 1
                    rew_file.write("Episode: {}, Reward: {} \n".format(num_episodes, info['episode']['r']))
                    all_rewards.append(info['episode']['r'])
            
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            
            ## Attack obs (if any)
            old_obs = obs.clone()
            if args.attacker:
                if args.attacker == "sarl":
                    obs = attacker.attack_stoc(obs, recurrent_hidden_states, masks, 
                            epsilon=args.epsilon, device=device)
                elif args.attacker == "random":
                    obs = attacker.attack_stoc(obs, epsilon=args.epsilon, fgsm=args.fgsm, 
                            device=device)
                else:
                    obs = attacker.attack_stoc(actor_critic, obs, recurrent_hidden_states, 
                            masks, epsilon=args.epsilon, fgsm=args.fgsm, lr=args.attack_lr, 
                            pgd_steps=args.attack_steps, device=device, rand_init=args.rand_init,
                            momentum=args.momentum)

            # print(old_obs-obs)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        rollouts.after_update()
        
        if num_episodes >= args.test_episodes:
            break

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Iteration {}, num timesteps {}, FPS {}"
                .format(j, total_num_steps, int(total_num_steps / (end - start))))
            if len(episode_rewards) > 1:
                print(
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            if args.attacker:
                print("attack amount", torch.norm(obs-old_obs, p=np.inf))
    
    all_rewards = np.array(all_rewards)
    print("Average rewards", np.mean(all_rewards).round(2), "std", np.std(all_rewards).round(2))
    rew_file.write("Average rewards:" + str(np.mean(all_rewards).round(2)) + ", std:" + str(np.std(all_rewards).round(2)))
    rew_file.close()

if __name__ == "__main__":
    main()
