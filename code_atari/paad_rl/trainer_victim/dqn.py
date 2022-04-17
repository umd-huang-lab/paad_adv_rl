import sys
import pickle
import numpy as np
from collections import namedtuple, deque
import time
from paad_rl.utils.monitor import Monitor
from paad_rl.utils.dqn_core import *
from paad_rl.utils.atari_utils import *
from paad_rl.utils.replay_buffer import *
from paad_rl.utils.schedule import *
from paad_rl.utils.param import Param
import random
import gym
import gym.spaces
gym.logger.set_level(40)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def dqn(
    env_name,
    exploration,
    trained_dir=None,
    frame_total=int(3e+6),
    replay_buffer_size=int(5e+5),
    batch_size=256,
    lr=0.0025,
    gamma=0.99,
    update_freq = None,
    tau=1e-3,
    learning_starts=100,
    learning_freq=4,
    max_steps=10000,
    log_steps=100,
    exp_name="PongNoFrameskip-v4",
    seed=0,
    doubleQ=False,
    duel=False,
    prioritized_replay=False,
    beta_schedule = None):

    """
    env_name (str):
        name of the environment
    exploration (Schedule Object):
        Schedule of the epsilon in the epsilon-greedy exploration
        At time step t, calling exploration.value(t) will return the 
        scheduled epsilon
    trained_dir (str):
        If not None, the training script will load the halfly trained model 
        from the given directory and continue the training process. 
    frame_total (int):
        The total amount of frames during training
    replay_buffer_size(int):
        size of the replay buffer
    batch_size (int):
        size of the batch
    lr (float):
        learning rate of the Q agent
    update_freq (int):
        It decides how often does the target Q network gets updated. Default is 2500, the same 
        as the one used in the original Deepmind paper
    tau (float):
        If update_freq is None, then we use a soft Polyak updates. 
        Q_target = (1-tau)*Q_target+tau*Q_current
    learning_starts (int): 
        It decides how many environment steps after which we start Q learning process.
    learning_freq (int): 
        It decides how many environment steps between every Q learning update.
    max_steps (int):
        During rollout, it decides the maximum amount of the environment steps.
    log_steps (int):
        It decides how often does the learning process of Q-learning agent gets logged,
        during which the latest Q-learning agent will also be saved.
    duel, doubleQ, prioritized_replay (bool):
        Whether we use dueling dqn, double dqn, ot priortized experience replay
    """
    dtype = Param.dtype
    device = Param.device
    
    env = gym.make(env_name)
    env = Monitor(env)
    env = make_env(env, frame_stack=True, scale=False)
    q_func = model_get('Atari', num_actions = env.action_space.n, duel=duel)
        
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    num_actions = env.action_space.n  
    
    dqn_agent = DQN_Agent(q_func, learning_rate=lr,doubleQ=doubleQ, atari=True,update_freq=update_freq)
    if trained_dir is not None:
        dqn_agent.load_state_dict(torch.load(trained_dir, map_location=Param.device))
    
    if not prioritized_replay:
        replay_buffer = DQNReplayBuffer(num_actions,replay_buffer_size,batch_size,seed)
    else:
        replay_buffer = DQNPrioritizedBuffer(replay_buffer_size,batch_size=batch_size,seed=seed)
    
    ### Set up logger file
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(exp_name)), "wt")
    
    ### Prepareing for running on the environment
    t, counter = 0, 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    scores = []
    scores_window = deque(maxlen=log_steps)
    ep_len_window = deque(maxlen=log_steps)
    time_window = deque(maxlen=log_steps)
    
    while (counter<frame_total):
        t += 1
        score, steps = 0, 0
        last_obs = env.reset()
        start = time.time()
        while (True):
            if (counter>learning_starts or trained_dir is not None):
                ### Epsilon Greedy Policy
                last_obs_normalized = last_obs/255. 
                eps = exploration.value(counter)
                action = dqn_agent.select_epilson_greedy_action(last_obs_normalized, eps)
            else: 
                ### Randomly Select an action before the learning starts
                action = random.randrange(num_actions)
            # Advance one step
            obs, reward, done, info = env.step(action)
            steps += 1
            counter+=1
            score += reward
            ### Add the experience into the buffer
            replay_buffer.add(last_obs, action, reward, obs, done)
            ### Update last observation
            last_obs = obs
            
            ### Q learning udpates
            if (counter > learning_starts and
                    counter % learning_freq == 0):
                if not prioritized_replay:
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample()
                    not_done_mask = 1 - done_mask
                    obs_batch = obs_batch/255. 
                    next_obs_batch = next_obs_batch/255.
                    dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                     next_obs_batch, not_done_mask, gamma, tau)    
                else:
                    obs_batch, act_batch, rew_batch, next_obs_batch, \
                    done_mask, indices, weights = replay_buffer.sample(beta=beta_schedule.value(counter))
                    obs_batch, next_obs_batch = obs_batch.squeeze(1), next_obs_batch.squeeze(1)
                    not_done_mask = (1 - done_mask).unsqueeze(1)
                    obs_batch = obs_batch/255. 
                    next_obs_batch = next_obs_batch/255. 
                    priority = dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                     next_obs_batch, not_done_mask, gamma, tau, weights)   
                    replay_buffer.update_priorities(indices, priority.cpu().numpy())
            if done or steps>max_steps:
                ep_len_window.append(steps)
                scores_window.append(score)
                steps = 0
                break
         
        scores.append(score)
        time_window.append(time.time()-start)
        
        ### print and log the learning process
        if t % log_steps == 0 and counter>learning_starts:
            print("------------------------------Episode {}------------------------------------".format(t))
            logger_file.write("------------------------------Episode {}------------------------------------\n".format(t))
            print('Num of Interactions with Environment:{:.2f}k'.format(counter/1000))
            logger_file.write('Num of Interactions with Environment:{:.2f}k\n'.format(counter/1000))
            print('Mean Training Reward per episode: {:.2f}'.format(np.mean(scores_window)))
            logger_file.write('Mean Training Reward per episode: {:.2f}\n'.format(np.mean(scores_window)))
            print('Average Episode Length: {:.2f}'.format(np.mean(ep_len_window)))
            logger_file.write('Average Episode Length: {:.2f}\n'.format(np.mean(ep_len_window)))
            print('Average Time: {:.2f}'.format(np.mean(time_window)))
            logger_file.write('Average Time: {:.2f}\n'.format(np.mean(time_window)))
            eval_reward = roll_out_atari(dqn_agent, env)
            print('Eval Reward:{:.2f}'.format(eval_reward))
            logger_file.write('Eval Reward:{:.2f}\n'.format(eval_reward))
            logger_file.flush()
            dqn_agent.save(exp_name=exp_name)
    logger_file.close()
    dqn_agent.save(exp_name=exp_name)
    return 
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-no_cuda', action="store_true")
    parser.add_argument('-cuda', '--which_cuda', type=int, default=1)
    
    parser.add_argument('--trained_dir', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--frame_total', type=int, default=3e+6)
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--exp_initp', type=float, default=1.0)
    parser.add_argument('--exp_finalp', type=float, default=0.05)
    
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('-weight_decay', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=2500)
    
    parser.add_argument('-doubleQ', action="store_true")
    parser.add_argument('-duel', action='store_true')
    
    parser.add_argument('--learning_starts', type=int, default=50000)
    parser.add_argument('--learning_freq', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('-prioritized_replay', action='store_true')
    parser.add_argument('--beta_start', type=float, default=0.4)
    parser.add_argument('--beta_steps', type=int, default=1e+6)
    parser.add_argument('--exploration_frames', type=int, default=1e+6)
    
    args = parser.parse_args()
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(args.which_cuda)))
    
    if args.prioritized_replay:
        print("Prioritized Experience Replay")
        beta_schedule = PiecewiseSchedule([(0,args.beta_start), (args.frame_total, 1.0)], 
                                                outside_value=1.0)
    else:
        beta_schedule = None
    
    ### Setup Exploration Schedule
    exploration = PiecewiseSchedule([(0, args.exp_initp), (args.exploration_frames, args.exp_finalp)], 
                                    outside_value=args.exp_finalp)
    dqn(env_name=args.env,
    trained_dir = args.trained_dir,
    exploration=exploration,
    frame_total=args.frame_total,
    replay_buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    lr=args.lr,
    gamma=args.gamma,
    tau = args.tau,
    learning_starts = args.learning_starts,
    learning_freq=args.learning_freq,
    log_steps=args.log_steps,
    exp_name=args.exp_name,
    seed=args.seed,
    doubleQ=args.doubleQ,
    duel = args.duel, 
    update_freq = args.update_freq,
    prioritized_replay = args.prioritized_replay,
    beta_schedule = beta_schedule,
    max_steps=args.max_steps)
    
