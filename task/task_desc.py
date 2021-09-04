import gym
import pybulletgym
import torch

from utils.env_parse import make_env
from utils.env_parse import get_pybulletgym_env_list
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class GymTask:
    def __init__(self):
        self.env = None
        self.device = None
        self.num_envs = None
        self.observation_space = None
        self.state_space = None
        self.action_space = None

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class HopperBullet(GymTask):
    def __init__(self, num_envs, device):
        super().__init__()
        self.num_envs = num_envs
        env_list = get_pybulletgym_env_list()
        env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]

        if self.num_envs == 1:
            self.env = gym.make(env_name)
        else:
            self.env = DummyVecEnv([make_env(env_id=env_name, rank=i) for i in range(self.num_envs)])
            # self.env = SubprocVecEnv([make_env(env_id=env_name, rank=i) for i in range(self.num_envs)], start_method='fork')

        self.device = device
        self.observation_space = self.env.observation_space
        self.state_space = self.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        obs = self.env.reset()
        return torch.FloatTensor(obs).view(self.num_envs, -1).to(self.device)

    def step(self, action):
        if len(action.shape) > 1:
            action = action.squeeze(0)
        _next_obs, _rew, _done, info = self.env.step(action)

        next_obs = torch.FloatTensor(_next_obs).view(self.num_envs, -1).to(self.device)
        rew = torch.FloatTensor([_rew]).squeeze(0).to(self.device)
        done = torch.FloatTensor([_done]).to(self.device)

        # for f in info:
        #     if bool(f):
        #         for k, v in f.items():
        #             f[k] = torch.Tensor([v]).to(self.device)
        return next_obs, rew, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
