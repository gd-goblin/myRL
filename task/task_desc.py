import gym
import pybulletgym
import torch
import numpy as np

from utils.env_parse import make_env, make_custom_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack


def env_parser(env_name, rank=None):
    if type(env_name) is str and rank is not None:
        return make_env(env_id=env_name, rank=rank)
    elif type(env_name) is str and rank is None:
        return gym.make(env_name)
    elif type(env_name) is not str and rank is None:
        return make_custom_env(env_name, 0)
    else:
        return make_custom_env(env_name, rank)


class WrapperVecEnv(gym.Env):
    def __init__(self, env_name, num_envs, device, normalized_env=True):
        super().__init__()
        self.num_envs = num_envs
        self.device = device
        self.normalized_env = normalized_env

        if self.num_envs == 1:
            # self.venv = gym.make(env_name)
            e = env_parser(env_name)
            print(e)
            print(e.observation_space)
            self.venv = DummyVecEnv([lambda: env_parser(env_name)])
        else:
            # self.env = DummyVecEnv([make_env(env_id=env_name, rank=i) for i in range(self.num_envs)])
            self.venv = SubprocVecEnv([env_parser(env_name, i) for i in range(self.num_envs)])

        if self.normalized_env:
            self.venv = VecNormalize(self.venv)

        self.low = torch.Tensor(self.venv.action_space.low).to(self.device)
        self.high = torch.Tensor(self.venv.action_space.high).to(self.device)

        self.observation_space = self.venv.observation_space
        self.state_space = self.observation_space
        self.action_space = self.venv.action_space

    def rescale_action(self, action):
        return self.low + (0.5 * (action + 1.0) * (self.high - self.low))

    def reset(self):
        obs = self.venv.reset()
        return torch.FloatTensor(obs).view(self.num_envs, -1).to(self.device)

    def step(self, action):
        action = self.rescale_action(action)
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy()
        _next_obs, _rew, _done, info = self.venv.step(action)

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

    def save(self, path):
        return self.venv.save(path)

    def load(self, path):
        self.venv = VecNormalize.load(load_path=path, venv=self.venv)
        self.venv.training = False
        self.venv.norm_reward = False

    def render(self):
        return self.venv.render()

    def close(self):
        return self.venv.close()
