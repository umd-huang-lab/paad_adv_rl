from paad_rl.utils.param import Param
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical
import copy

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)
    
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    obs_buf, act_buf, adv_buf, rew_buf, ret_buf, val_buf: numpy array
    logp_array: torch tensor
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf  = np.zeros(combined_shape(size, obs_dim))
        self.act_buf  = np.zeros(combined_shape(size, act_dim))
        self.adv_buf  = np.zeros(size)
        self.rew_buf  = np.zeros(size)
        self.ret_buf  = np.zeros(size)
        self.val_buf  = np.zeros(size)
        self.logp_buf = torch.zeros(size).to(Param.device).type(Param.dtype)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf, 0), np.std(self.adv_buf, 0)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=from_numpy(self.obs_buf), act=from_numpy(self.act_buf), ret=from_numpy(self.ret_buf),
                    adv=from_numpy(self.adv_buf), logp=self.logp_buf)
        return data

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

### Clip the Action
def clip(action, low, high):
    return np.clip(action, low, high)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class Shared_CNN:
    def __init__(self,num_actions=4):
        self.n_actions = num_actions
        Shared_CNN.cnn_layers = CNN_Layers(num_actions=num_actions)
    def shared_cnn_layers(self):
        return Shared_CNN.cnn_layers
    
class CNN_Layers(nn.Module):
    def __init__(self, in_channels=1, num_actions=18):
        super(CNN_Layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

    def forward(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return x
    
class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, atari=False):
        super().__init__()
        if atari:
            Shared_CNN(num_actions=act_dim)
        self.cnn_layers = Shared_CNN.cnn_layers if atari else None
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if (self.cnn_layers is not None):
            obs = self.cnn_layers(obs)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, atari=False):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        if atari:
            Shared_CNN(num_actions=act_dim)
        self.cnn_layers = Shared_CNN.cnn_layers if atari else None
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if (self.cnn_layers is not None):
            obs = self.cnn_layers(obs)
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPBetaActor(Actor):
    
    ### Beta distribution, dealing with the case where action is bounded in the 
    ### box (-epsilon, epsilon)
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, high, atari=False):
        super().__init__()
        self.high = high
        if atari:
            Shared_CNN(num_actions=act_dim)
        self.cnn_layers = Shared_CNN.cnn_layers if atari else None
        self.alpha = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.beta = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        if (self.cnn_layers is not None):
            obs = self.cnn_layers(obs)
        alpha = self.alpha(obs)
        beta  = self.beta(obs)
        alpha = torch.log(1+torch.exp(alpha))+1
        beta  = torch.log(1+torch.exp(beta))+1
        return Beta(alpha, beta)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, atari=False):
        super().__init__()
        self.cnn_layers = Shared_CNN.cnn_layers if atari else None
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        if (self.cnn_layers is not None):
            obs = self.cnn_layers(obs)
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape

class RandomActorCritic():

    def __init__(self, env):
        super().__init__()
        self.act_dim = env.action_space.shape[0]
        self.low, self.high = env.action_space.low, env.action_space.high
    def step(self, obs):
        return np.random.uniform(self.low, self.high, (self.act_dim,)), None, None
    
    

class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh, 
                 beta=False, atari=False):
        super().__init__()
    
        obs_dim = observation_space.shape[0] if not atari else 512
        self.high = torch.from_numpy(action_space.high).type(Param.dtype).to(Param.device)
        self.beta = beta ### Whether to use beta distribution to deal with clipped action space
        # policy builder depends on action space
        if isinstance(action_space, Box) and not beta:
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, atari=atari)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, atari=atari)
        else:
            self.pi = MLPBetaActor(obs_dim, action_space.shape[0], hidden_sizes, activation, self.high, atari=atari)
        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation, atari=atari)
        
        self.MovingMeanStd = MovingMeanStd(observation_space.shape)
        self.moving_mean = torch.zeros(observation_space.shape).to(Param.device).type(Param.dtype)
        self.moving_std  = torch.ones(observation_space.shape).to(Param.device).type(Param.dtype)
        
    def step(self, obs, train=False):
        with torch.no_grad():
            if train:
                self.MovingMeanStd.push(obs)
                self.moving_mean = self.MovingMeanStd.mean()
                self.moving_std  = self.MovingMeanStd.std()
            obs = (obs - self.moving_mean)/(self.moving_std+1e-6)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            if self.beta:
                a = 2*a*self.high-self.high ### Clip to the correct range
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]
    
    def save(self, log_dir= os.path.join(Param.model_dir,r'./ppo'), model_name='ppo_policy'):
        if (not self.beta):
            torch.save(self.state_dict(), os.path.join(log_dir,model_name))
        else:
            log_dir = Param.adv_dir
            torch.save(self.state_dict(), os.path.join(log_dir,model_name))
    
    ### Return Normalized Observation
    def normalize(self, obs):
        return (obs - self.moving_mean)/self.moving_std

def rollout(agent, env_fn, num_trajectories=10, num_steps=1000, render=False):
    env = env_fn()
    rews = []
    for i in range(num_trajectories):
        o = env.reset()
        total_rew = 0
        for t in range(num_steps):
            a = agent.act(torch.from_numpy(o).to(Param.device).type(Param.dtype))
            (o, reward, done, _info) = env.step(a.cpu().numpy())
            total_rew += reward
            if render: 
                env.render()
            if done: break
        rews.append(total_rew)
    return sum(rews)/len(rews)

### Calculating moving meana and standard deviation
class MovingMeanStd:

    def __init__(self, shape):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.shape = shape

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        if self.n > 0:
            return self.new_m
        else:
            return torch.zeros(self.shape).to(Param.device).type(Param.dtype)

    def variance(self):
        if self.n > 1:
            return self.new_s / (self.n - 1) 
        else:
            return torch.ones(self.shape).to(Param.device).type(Param.dtype)

    def std(self):
        return torch.sqrt(self.variance())
