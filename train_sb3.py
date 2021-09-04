import gym
import pybulletgym
from utils.env_parse import make_env
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

env_name = 'HopperPyBulletEnv-v0'  # 'FetchPickAndPlace-v1'
env = make_vec_env(env_id=env_name, n_envs=2, vec_env_cls=DummyVecEnv)
n_procs = 2
total_procs = 1
env = DummyVecEnv([make_env(env_id=env_name, rank=i+total_procs) for i in range(n_procs)])
print("env: ", env)
obs = env.reset()
print("obs: ", obs.shape)

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
