import torch
import time
import pybullet as p
from utils.utils import *
from task.task_desc import WrapperVecEnv
from utils.env_parse import get_pybulletgym_env_list
from utils.env_parse import make_env
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def train_test():
    env_list = get_pybulletgym_env_list()
    env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    env = WrapperVecEnv(env_name=env_name, num_envs=8, device=device, normalized_env=True)
    # n_procs = 2
    # env = DummyVecEnv([make_env(env_id=env_name, rank=i) for i in range(n_procs)])
    # env = VecNormalize(env)
    print("env: ", env)
    obs = env.reset()
    low, high = env.action_space.low, env.action_space.high
    print("low: {}, high: {}".format(low, high))
    print("obs: ", obs.shape)
    for i in range(10):
        action = env.sample_action()
        print("action min/max: {} / {}".format(action.min(), action.max()))
        obs, rew, _, _ = env.step(action)
        print("obs min/max: {} / {}".format(obs.min(), obs.max()))

    env.close()
    exit()
    env = gym.make(env_name)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=5000)

    env.render()
    obs = env.reset()
    for i in range(5000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        if done:
            obs = env.reset()


from task.mirobot import Mirobot
import operator


def custom_task():
    print("custom task test")
    p.connect(p.GUI)
    env = Mirobot()
    obs = env.reset(p)
    for i in range(100):
        time.sleep(0.1)
        print("obs: ", rad2deg(obs), obs.shape)
        pass


if __name__ == '__main__':
    # train_test()
    custom_task()
