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
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

from paad_rl.a2c_ppo_acktr import algo, utils
from paad_rl.a2c_ppo_acktr.algo import gail
from paad_rl.a2c_ppo_acktr.arguments import get_args
from paad_rl.a2c_ppo_acktr.envs import make_vec_envs
from paad_rl.a2c_ppo_acktr.model import Policy, BetaMLP
from paad_rl.a2c_ppo_acktr.storage import RolloutStorage
from paad_rl.utils.ppo_core import mlp
COEFF = 1

def get_policy(victim, obs, recurrent, masks, cont):
    if cont:
        return victim.get_dist(obs, recurrent, masks).mean
    else:
        return victim.get_dist(obs, recurrent, masks).probs

def obs_dir_perturb_fgsm(victim, obs, recurrent, masks, direction, epsilon, device, cont=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    clean_policy = get_policy(victim, obs, recurrent, masks, cont).detach()
    policy = get_policy(victim, obs+perturb, recurrent, masks, cont)
    diff = policy - clean_policy
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 

    
    loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))

    loss.backward()
    grad = perturb.grad.data
    perturb.data -= epsilon * torch.sign(grad)

    return perturb.detach()

def obs_dir_perturb_momentum(victim, obs, recurrent, masks, direction, epsilon, device, cont=False,
    maxiter=10):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    clean_policy = get_policy(victim, obs, recurrent, masks, cont).detach()
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 
        
    def loss_fn(perturbed_obs):
        perturbed_policy = get_policy(victim, perturbed_obs, recurrent, masks, cont)
        diff = perturbed_policy - clean_policy
        loss = torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
        return loss
    
    mu = 0.5
    v = torch.zeros_like(obs).to(device)
    lr = epsilon / maxiter

    obs_adv = obs.clone().detach().to(device)
    for i in range(maxiter):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv + mu * v)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad

        v = mu * v + gradients/torch.norm(gradients, p=1)
        obs_adv += v.sign().detach() * lr

        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)
       
    return obs_adv.detach() - obs.detach()


def obs_dir_perturb_pgd(victim, obs, recurrent, masks, direction, epsilon, device, cont=False,
    maxiter=10, lr=1e-4, etol=1e-7, rand_init=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    clean_policy = get_policy(victim, obs, recurrent, masks, cont).detach()
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 
    
    obs_adv = obs.clone().detach().to(device)
    if rand_init:
        obs_adv += (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    for i in range(maxiter):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        policy = get_policy(victim, _obs_adv, recurrent, masks, cont)
        diff = policy - clean_policy
        loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
        loss.backward()

        gradients = _obs_adv.grad.sign().detach()
        obs_adv -= gradients * lr
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)

    return obs_adv.detach() - obs.detach()
    


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
    
    if envs.action_space.__class__.__name__ == "Discrete":
        action_space = Box(-1.0, 1.0, (envs.action_space.n-1,))
        cont = False
    elif envs.action_space.__class__.__name__ == "Box":
        print("action space", envs.action_space.shape[0])
        action_space = Box(-1.0, 1.0, (envs.action_space.shape[0],))
        cont = True

    #### PATHs ####
    optim_method = "momentum" if args.momentum else ("fgsm" if args.fgsm else "pgd" )
    vtype = "_vdet" if args.v_det else ""
    exp_name = "obspol_attacker_{}_e{}_{}{}".format(args.env_name, args.epsilon, optim_method, vtype)
    
    model_dir = os.path.join(args.adv_dir, args.algo)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    victim_dir = args.victim_dir

    model_path = os.path.join(model_dir, exp_name)

    result_path = os.path.join(args.res_dir, exp_name + ".txt")
    rew_file = open(result_path, "wt")
    
    use_beta = args.beta

    is_image = len(envs.observation_space.shape) > 1
    
    if is_image:
        obs_dim = envs.observation_space.shape
    else:
        obs_dim = envs.observation_space.shape[0]

    if envs.action_space.__class__.__name__ == "Discrete":
        act_dim = envs.action_space.n
    elif envs.action_space.__class__.__name__ == "Box":
        act_dim = envs.action_space.shape[0]

    actor_critic = Policy(
        envs.observation_space.shape,
        action_space,
        beta=use_beta,
        epsilon=args.epsilon,
        base_kwargs={'recurrent': args.recurrent_policy})
    
    if args.load:
        old_steps, load_states, _ = torch.load(model_path)
        actor_critic.load_state_dict(load_states)
        print("load a model trained for", old_steps, "steps")
    
    actor_critic = actor_critic.to(device)

    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            beta=use_beta)
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
            beta=use_beta)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True, beta=use_beta)
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    ## load pre-trained victim
    ## load victim
    victim = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
    agent_states, ob_rms = torch.load(os.path.join(victim_dir, args.env_name), map_location=device)
    print("loaded victim model from", os.path.join(victim_dir, args.env_name))
    victim.load_state_dict(agent_states)
    victim.to(device)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    victim_recurrent = torch.zeros(
        args.num_processes, victim.recurrent_hidden_state_size, device=device)
    
    default_recurrent = torch.zeros(
        args.num_processes, victim.recurrent_hidden_state_size, device=device)
    default_masks = torch.ones(args.num_processes, 1, device=device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    

    best_performance = np.inf
    performance_record = deque(maxlen=20)
    

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
                    rollouts.masks[step], beta=use_beta, deterministic=args.det)

            # the action is the policy perturbation
            if not cont:
                perturb_direction = torch.cat((action, -torch.sum(action, dim=1, keepdim=True)), 1)
                perturb_direction /= torch.norm(perturb_direction)
            else:
                perturb_direction = action
                
            if args.fgsm:
                obs_perturb = obs_dir_perturb_fgsm(victim, rollouts.obs[step], victim_recurrent,
                        rollouts.masks[step], perturb_direction, args.epsilon, device, cont)
            elif args.momentum:
                obs_perturb = obs_dir_perturb_momentum(victim, rollouts.obs[step], victim_recurrent,
                        rollouts.masks[step], perturb_direction, args.epsilon, device, cont,
                        maxiter=args.attack_steps)
            else:
                obs_perturb = obs_dir_perturb_pgd(victim, rollouts.obs[step], victim_recurrent,
                        rollouts.masks[step], perturb_direction, args.epsilon, device, cont,
                        lr=args.attack_lr, maxiter=args.attack_steps, rand_init=args.rand_init)

            _, v_action, _, victim_recurrent = victim.act(
                        rollouts.obs[step]+obs_perturb, victim_recurrent,
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
            # save if the current performance is better the historical best, otherwise do not overwrite
            if args.train_nn or (len(performance_record) > 1 and np.mean(performance_record) < best_performance):
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                print("*** save for", np.mean(performance_record))
                best_performance = np.mean(performance_record)
                
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
            if len(episode_rewards) > 1 and not args.train_nn:
                print(
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, {:.1f}, {:.1f}, {:.1f}"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), value_loss, action_loss, dist_entropy))
                rew_file.write("Step: {}, Reward: {} \n".format(total_num_steps, np.mean(episode_rewards)))
            # print(obs_perturb[0])
            print("perturb norm", torch.norm(obs_perturb, p=np.inf))

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
