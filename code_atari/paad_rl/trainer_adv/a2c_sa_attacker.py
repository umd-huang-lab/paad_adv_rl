import copy
import glob
import os
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
    action_space = Box(-args.epsilon, args.epsilon, envs.observation_space.shape)
    # print("high", action_space.high, "low", action_space.low)



    #### PATHs ####
    vtype = "_vdet" if args.v_det else ""
    exp_name = "obs_attacker_{}_e{}_{}{}".format(args.env_name, args.epsilon,
                "fgsm" if args.fgsm else "pgd", vtype)
    model_dir = os.path.join(args.adv_dir, args.algo)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    victim_dir = args.victim_dir

    model_path = os.path.join(model_dir, exp_name)

    result_path = os.path.join(args.res_dir, exp_name + ".txt")

    beta = False
    
    actor_critic = Policy(
        envs.observation_space.shape,
        action_space,
        beta=beta,
        epsilon=args.epsilon,
        base_kwargs={'recurrent': args.recurrent_policy})
    if args.load:
        old_steps, load_states, _ = torch.load(model_path)
        actor_critic.load_state_dict(load_states)
        print("load a model trained for", old_steps, "steps")
    actor_critic.to(device)
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            beta=beta)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            beta=beta)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True, beta=beta)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    ## load pre-trained learner
    victim = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    agent_states, ob_rms = torch.load(os.path.join(victim_dir, args.env_name), map_location=device)
    victim.load_state_dict(agent_states)
    victim.to(device)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    
    victim_recurrent = torch.zeros(
        args.num_processes, victim.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    best_performance = np.inf
    performance_record = deque(maxlen=20)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    rew_file = open(result_path, "wt")

    rewards = torch.zeros(args.num_processes, 1, device=device)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], beta=beta)
            if len(action_space.high.shape) > 1:
                perturb = action.view(-1, action_space.high.shape[0], 
                        action_space.high.shape[1], action_space.high.shape[2])
                perturb = perturb.clamp(-args.epsilon, args.epsilon) #perturb * 2 * args.epsilon - args.epsilon
            else:
                perturb = action.clamp(-args.epsilon, args.epsilon) #action * 2 * args.epsilon - args.epsilon
            # the action is the state perturbation to the victim
            _, v_action, _, victim_recurrent = victim.act(
                    rollouts.obs[step]+perturb, victim_recurrent,
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(v_action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    performance_record.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
                 
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, -reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if j > 50 and args.save_interval > 0 and (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            if len(performance_record) > 1 and np.mean(performance_record) < best_performance:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                print("*** save for", np.mean(performance_record))
                best_performance = np.mean(performance_record)
#                 save_path = os.path.join(args.save_dir, args.algo)
#                 try:
#                     os.makedirs(save_path)
#                 except OSError:
#                     pass

                torch.save([
                    total_num_steps,
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], model_path)

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
                rew_file.write("Step: {}, Reward: {} \n".format(total_num_steps, np.mean(episode_rewards)))
            print(perturb[0])
            print("norm", torch.norm(perturb, p=np.inf))
            
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
    
    modelfinal_path = os.path.join(model_dir, exp_name + "_final")
    torch.save([
        args.num_env_steps,
        actor_critic.state_dict(),
        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
    ], modelfinal_path)

    rew_file.close()

if __name__ == "__main__":
    main()
