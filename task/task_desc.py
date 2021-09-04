import gym
import pybulletgym
import torch
import numpy as np

from utils.env_parse import make_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack


class WrapperVecEnv(gym.Env):
    def __init__(self, env_name, num_envs, device, normalized_env=True):
        super().__init__()
        self.num_envs = num_envs
        self.device = device

        if self.num_envs == 1:
            self.env = gym.make(env_name)
        else:
            # self.env = DummyVecEnv([make_env(env_id=env_name, rank=i) for i in range(self.num_envs)])
            self.env = SubprocVecEnv([make_env(env_id=env_name, rank=i) for i in range(self.num_envs)])
            if normalized_env:
                self.env = VecNormalize(self.env)

        self.low = torch.Tensor(self.env.action_space.low).to(self.device)
        self.high = torch.Tensor(self.env.action_space.high).to(self.device)

        self.observation_space = self.env.observation_space
        self.state_space = self.observation_space
        self.action_space = self.env.action_space

    def rescale_action(self, action):
        return self.low + (0.5 * (action + 1.0) * (self.high - self.low))

    def reset(self):
        obs = self.env.reset()
        return torch.FloatTensor(obs).view(self.num_envs, -1).to(self.device)

    def step(self, action):
        action = self.rescale_action(action)
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy()
        _next_obs, _rew, _done, info = self.env.step(action)

        next_obs = torch.FloatTensor(_next_obs).view(self.num_envs, -1).to(self.device)
        rew = torch.FloatTensor([_rew]).squeeze(0).to(self.device)
        done = torch.FloatTensor([_done]).to(self.device)

        # for f in info:
        #     if bool(f):
        #         for k, v in f.items():
        #             f[k] = torch.Tensor([v]).to(self.device)
        return next_obs, rew, done, info

    def sample_action(self):
        action = torch.from_numpy(self.action_space.sample()).to(self.device)
        return action.repeat(self.num_envs, 1)  # same action for all envs

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
