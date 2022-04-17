import copy
import glob
import os
import time
import sys
import random
from collections import deque

import gym
from gym.spaces.box import Box
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.atla_args import get_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
from paad_rl.a2c_ppo_acktr.model import Policy, BetaMLP
# from paad_rl.evaluation import evaluate
from paad_rl.utils.ppo_core import mlp
from paad_rl.utils.schedule import LinearScheduler

from paad_rl.attacker.attacker import *
# from paad_rl.attacker.pa_obs_attacker import PA_Obs_Attacker
from paad_rl.trainer_adv.a2c_pa_attacker import get_policy, obs_dir_perturb_fgsm, obs_dir_perturb_pgd

def collect_trajectory(args, actor_critic, attacker, naive_attacker, rollouts, envs, episode_rewards, eps):
    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            
        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                # print("epi", info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]).to(device)
        bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos]).to(device)
        
        # apply optimal attack
        if args.adv_ratio >= random.random():
            obs = attacker.attack_stoc(actor_critic, obs, recurrent_hidden_states, 
                    masks, eps, fgsm=args.fgsm, lr=args.attack_lr, 
                    pgd_steps=args.attack_steps, device=args.device)

        # apply naive attack
        else:
            obs = naive_attacker.attack_stoc(actor_critic, obs, recurrent_hidden_states, 
                    masks, eps, fgsm=args.fgsm, lr=args.attack_lr, 
                    pgd_steps=args.attack_steps, device=args.device)


        rollouts.insert(obs, recurrent_hidden_states, action,
                action_log_prob, value, reward, masks, bad_masks)
        
    return rollouts


def collect_adv_trajectory(args, attacker, victim, v_recurrent, adv_rollouts, adv_envs, adv_episode_rewards, performance_record, eps):
    victim_recurrent = v_recurrent
    for step in range(args.num_steps):
        # Sample adversary actions (attack)
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = attacker.adv_policy.act(
                    adv_rollouts.obs[step], adv_rollouts.recurrent_hidden_states[step],
                    adv_rollouts.masks[step], beta=args.beta, deterministic=args.det)

        # the action is the policy perturbation
        if not args.cont:
            perturb_direction = torch.cat((action, -torch.sum(action, dim=1, keepdim=True)), 1)
        else:
            perturb_direction = action

        if args.use_nn:
            if args.is_image:
                flat_obs = adv_rollouts.obs[step].view(-1, args.obs_dim)
                flat_perturb = attacker.obs_attacker.perturb_batch(flat_obs,
                        perturb_direction)
                obs_perturb = flat_perturb.view(-1, adv_envs.observation_space.shape[0], 
                        adv_envs.observation_space.shape[1], adv_envs.observation_space.shape[2])
            else:
                obs_perturb = attacker.obs_attacker.perturb_batch(adv_rollouts.obs[step],
                        perturb_direction)
        elif args.fgsm:
            obs_perturb = obs_dir_perturb_fgsm(victim, adv_rollouts.obs[step], victim_recurrent,
                    adv_rollouts.masks[step], perturb_direction, eps, args.device, args.cont)
        else:
            obs_perturb = obs_dir_perturb_pgd(victim, adv_rollouts.obs[step], victim_recurrent,
                    adv_rollouts.masks[step], perturb_direction, eps, args.device, args.cont,
                    lr=args.attack_lr, maxiter=args.attack_steps)

        with torch.no_grad():
            _, v_action, _, victim_recurrent = victim.act(
                    adv_rollouts.obs[step]+obs_perturb, victim_recurrent,
                    adv_rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = adv_envs.step(v_action)

        for info in infos:
            if 'episode' in info.keys():
                adv_episode_rewards.append(info['episode']['r'])
                performance_record.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]).to(device)
        bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos]).to(device)
            
        adv_rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, -reward, masks, bad_masks)

    return victim_recurrent, adv_rollouts

def env_info(envs):
    is_image = len(envs.observation_space.shape) > 1

    if is_image:
        obs_dim = 1
        for s in envs.observation_space.shape:
            obs_dim *= s
    else:
        obs_dim = envs.observation_space.shape[0]

    if envs.action_space.__class__.__name__ == "Discrete":
        act_dim = envs.action_space.n
    elif envs.action_space.__class__.__name__ == "Box":
        act_dim = envs.action_space.shape[0]

    return is_image, obs_dim, act_dim

def main():
    # set adv env
    # adv agent
    # adv rollouts
    # attack method
    # adv log dir

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    eval_adv_log_dir = log_dir + "_adv_eval"

    result_path = log_dir + "reward.txt"
    adv_result_path = log_dir + "adv_reward.txt"
    rew_file = open(result_path, "wt")
    adv_rew_file = open(adv_result_path, "wt")

    # utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    utils.cleanup_log_dir(eval_adv_log_dir)

    torch.set_num_threads(1)
    global device
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
    args.device = device
    
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    # print("The observation space is", envs.observation_space)

    adv_envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval() # don't normalize

    adv_vec_norm = utils.get_vec_normalize(adv_envs)
    if adv_vec_norm is not None:
        adv_vec_norm.eval()


    if envs.action_space.__class__.__name__ == "Discrete":
        action_space = Box(-1.0, 1.0, (envs.action_space.n-1,))
        cont = False
    elif envs.action_space.__class__.__name__ == "Box":
        # print("action space", envs.action_space.shape[0])
        action_space = Box(-1.0, 1.0, (envs.action_space.shape[0],))
        cont = True

    use_beta = args.beta

    is_image, obs_dim, act_dim = env_info(envs)

    args.is_image = is_image
    args.obs_dim = obs_dim
    args.cont = cont

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': True})
    # load pretrained policy model (victim)
    if args.load:
        print("load pretrained policy model")
        actor_critic_state, _ = \
                torch.load(os.path.join("./learned_models/{}/{}".format(args.algo, args.env_name)))
        actor_critic.load_state_dict(actor_critic_state)
    actor_critic = actor_critic.to(device)

    
    adv_actor_critic = Policy(
        envs.observation_space.shape,
        action_space,
        beta=use_beta,
        epsilon=args.epsilon,
        base_kwargs={'recurrent': args.recurrent_policy})

    if args.use_nn:
        if args.nn_hiddens:
            hidden_sizes = list(args.nn_hiddens)
        else:
            hidden_sizes=[128,128,128]
        # hidden_sizes=[64,64]
        attacker_network = mlp([obs_dim+act_dim] + \
                hidden_sizes + [obs_dim], torch.nn.Tanh, torch.nn.Tanh)
        # attacker_network = BetaMLP(obs_dim, act_dim, epsilon=args.epsilon)
        
    # load pretrained adversary model (attacker)
    if args.load_adv:
        if args.attacker == "obspol":
            print("load pretrained observation-policy attacker")
            if args.use_nn:
                _, adv_actor_critic_state, attacker_network, _ = \
                        torch.load(os.path.join("./learned_models/{}/attacker_{}".format(args.algo, args.env_name)))
                
            else:
                _, adv_actor_critic_state, _ = \
                        torch.load(os.path.join("./learned_models/{}/attacker_{}".format(args.algo, args.env_name)))
            adv_actor_critic.load_state_dict(adv_actor_critic_state)
        else:
            print("Attacker not match.")
            
    adv_actor_critic = adv_actor_critic.to(device)

    # construct obs_attacker and attacker
    if args.use_nn:
        attacker_network = attacker_network.to(device)
        
        obs_attacker = PA_Obs_Attacker(attacker_network, obs_dim, act_dim, 
                norm=np.inf, epsilon=args.epsilon, lr=args.attack_lr,
                original_shape=(envs.observation_space.shape if is_image else None))
                
        attacker = ObsPol_Attack(adv_actor_critic, obs_attacker, det=args.det, cont=cont)
    else: 
        attacker = ObsPol_Attack(adv_actor_critic, None, det=args.det, cont=cont)

    naive_attacker = attacker
    
    # construct agent and adversary with 'a2c'
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
            
        adversary = algo.A2C_ACKTR(
            adv_actor_critic,
            args.value_loss_coef,
            args.adv_entropy_coef,
            lr=args.adv_lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    adv_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    # reset environment
    obs = envs.reset()

    default_recurrent = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    default_masks = torch.ones(args.num_processes, 1, device=device)
    # initial attack
    obs = attacker.attack_stoc(actor_critic, obs, default_recurrent,
                        default_masks, epsilon=args.epsilon, fgsm=args.fgsm, lr=args.attack_lr,
                        pgd_steps=args.attack_steps, device=device)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    victim_recurrent = torch.zeros(
            args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        
    obs = adv_envs.reset()
    adv_rollouts.obs[0].copy_(obs)
    adv_rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    adv_episode_rewards = deque(maxlen=10)
    rewards = torch.zeros(args.num_processes, 1, device=device)

    best_performance = np.inf
    performance_record = deque(maxlen=20)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    num_alters = int(num_updates // args.policy_update_steps)


    if args.use_nn:
        nn_steps = args.attack_steps
        steps_decay_every = int(num_updates / nn_steps)
        nn_losses = deque(maxlen=100)

    # record policy and attacker update times
    policy_updates = 0
    adv_updates = 0
    
    # eps scheduler
    eps_scheduler = LinearScheduler(args.epsilon, args.eps_scheduler_opts)
    eps_scheduler.set_epoch_length(1000)
    eps_scheduler.step_batch()
    
    # start alternative updates
    # update policy model for policy_update_steps
    # then update adversary model for adv_update_steps
    for j in range(num_alters):
        
        eps_scheduler.step_batch()
        current_eps = eps_scheduler.get_eps()
        # update agent policy
        for i in range(args.policy_update_steps):

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                        agent.optimizer, policy_updates, num_updates,
                        agent.optimizer.lr if args.algo == "acktr" else args.lr)

            rollouts = collect_trajectory(args, actor_critic, attacker, naive_attacker, rollouts, envs, episode_rewards, current_eps)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                        rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                        rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                    args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            # print("cur value loss", value_loss, "cur action loss", action_loss)

            rollouts.after_update()
            policy_updates += 1

        # update adversary policy
        for i in range(args.adv_update_steps):

            if args.use_linear_lr_decay:
                # decrease adversary learning rate linearly
                utils.update_linear_schedule(
                        adversary.optimizer, adv_updates, num_updates,
                        adversary.optimizer.lr if args.algo == "acktr" else args.adv_lr)

            # targeted = False
            victim_recurrent, adv_rollouts = collect_adv_trajectory(args, attacker, actor_critic, victim_recurrent, adv_rollouts, adv_envs, adv_episode_rewards, performance_record, current_eps)

            with torch.no_grad():
                adv_next_value = adv_actor_critic.get_value(
                        adv_rollouts.obs[-1], adv_rollouts.recurrent_hidden_states[-1],
                        adv_rollouts.masks[-1]).detach()

            adv_rollouts.compute_returns(adv_next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            if args.train_nn:
                nn_loss = obs_attacker.update_stoc(actor_critic, default_recurrent, default_masks, device,
                        num_iter=nn_steps, cont=cont)
                nn_losses.append(nn_loss)
                # print("mean nn loss", np.mean(nn_losses), "cur nn loss", nn_loss)
            else:
                adv_value_loss, adv_action_loss, adv_dist_entropy = adversary.update(adv_rollouts)
                # print("cur adv-value loss", adv_value_loss, "cur adv-action loss", adv_action_loss)
                if args.use_nn:
                    nn_loss = obs_attacker.update_stoc(actor_critic, default_recurrent, default_masks, device,
                            num_iter=nn_steps, cont=cont)
                    nn_losses.append(nn_loss)
                    # print("mean nn loss", np.mean(nn_losses), "cur nn loss", nn_loss)

            adv_rollouts.after_update()
            
            if args.use_nn:
                attacker = ObsPol_Attack(adv_actor_critic, obs_attacker, det=args.det, cont=cont)
            else: 
                attacker = ObsPol_Attack(adv_actor_critic, None, det=args.det, cont=cont)
            adv_updates += 1

        # save model
        # save policy model for every interval-th alternating iteration
        if args.save_interval > 0 and j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            
            torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_atla.pt"))
            
            # save adversary model for every interval-th alternating iteration
            if len(performance_record) > 1 and np.mean(performance_record) < best_performance:
                total_adv_num_steps = adv_updates * args.num_processes * args.num_steps
                print("*** save for performance under optimal attack: ", np.mean(performance_record))
                best_performance = np.mean(performance_record)

                # update naive_attacker
                naive_attacker = attacker

                save_adv_path = os.path.join(save_path, args.attacker)
                try:
                    os.makedirs(save_adv_path)
                except OSError:
                    pass
                if args.use_nn:
                    torch.save([
                            total_adv_num_steps,
                            adv_actor_critic,
                            attacker_network,
                            getattr(utils.get_vec_normalize(adv_envs), 'ob_rms', None)
                    ], os.path.join(save_adv_path, args.env_name + "_atla.pt"))
                else:
                    torch.save([
                            total_adv_num_steps,
                            adv_actor_critic,
                            getattr(utils.get_vec_normalize(adv_envs), 'ob_rms', None)
                    ], os.path.join(save_adv_path, args.env_name + "_atla.pt"))
                
        # logging for every interval-th alternating iteration
        if j % args.log_interval == 0 and len(episode_rewards) > 1 and len(adv_episode_rewards) > 1:
            total_num_steps = policy_updates * args.num_processes * args.num_steps
            total_adv_num_steps = adv_updates * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Policy Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}\n"
                .format(policy_updates, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy))
            rew_file.write("Step: {}, Policy Reward: {} \n".format(total_num_steps, np.mean(episode_rewards)))
            print(
                "Adversary Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {}\n"
                .format(adv_updates, total_adv_num_steps,
                        int(total_adv_num_steps / (end - start)),
                        len(adv_episode_rewards), np.mean(adv_episode_rewards),
                        np.median(adv_episode_rewards), np.min(adv_episode_rewards),
                        np.max(adv_episode_rewards), adv_dist_entropy))
            adv_rew_file.write("Step: {}, Adversary Reward: {} \n".format(total_adv_num_steps, np.mean(adv_episode_rewards)))

        # evaluating policy model for every interval-th alternating iteration
        if (args.eval_interval is not None and len(episode_rewards) > 1 
                and j % args.eval_interval == 0):
            ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, episodes=50)

    modelfinal_path = os.path.join(save_path, exp_name + "_atla_final" + ".pt")
    adv_modelfinal_path = os.path.join(save_adv_path, exp_name + "_atla_final" + ".pt")
    torch.save([
        args.num_env_steps,
        actor_critic,
        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
    ], modelfinal_path)

    if args.use_nn:
        torch.save([
            args.num_env_steps,
            adv_actor_critic,
            attacker_network,
            getattr(utils.get_vec_normalize(adv_envs), 'ob_rms', None)
        ], adv_modelfinal_path)
    else:
        torch.save([
            args.num_env_steps,
            adv_actor_critic,
            getattr(utils.get_vec_normalize(adv_envs), 'ob_rms', None)
        ], adv_modelfinal_path)

    rew_file.close()


if __name__ == "__main__":
    main()
