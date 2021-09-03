from utils.env_parse import get_pybulletgym_env_list
import gym
import pybulletgym
import torch


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
    def __init__(self, device):
        super().__init__()
        env_list = get_pybulletgym_env_list()
        env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
        self.env = gym.make(env_name)
        self.num_envs = 1
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
        rew = torch.FloatTensor([_rew]).to(self.device)
        done = torch.FloatTensor([_done]).to(self.device)
        if bool(info):
            for k, v in info.items():
                info[k] = torch.Tensor([v]).to(self.device)
        return next_obs, rew, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
