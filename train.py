import torch
import os
import gym
import pybulletgym
from utils.env_parse import get_pybulletgym_env_list

import d3rlpy
from d3rlpy.algos import CQL
from algo.ppo import PPO, ActorCritic

from task.task_desc import WrapperVecEnv


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def render_test(env_name, n_frames):
    env = gym.make(env_name)
    env.render()
    env.reset()
    done = False
    for i in range(n_frames):
        if done:
            env.reset()
            print("stpes: {}, done: {}".format(i, done))

        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
    env.close()


def offline_train(env_name):
    dataset, env = d3rlpy.datasets.get_pybullet(env_name)

    cql = CQL(use_gpu=False)
    cql.fit(dataset,
            n_epochs=100,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer}
            )


def online_train(env_name, num_learning_iter, visualize=False):
    env = WrapperVecEnv(env_name=env_name, num_envs=16, device=device, normalized_env=True)
    env.render() if visualize else None
    log_dir = os.path.join('run', env_name)

    ppo = PPO(vec_env=env,
              learning_rate=1e-4,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=2048 // env.num_envs,
              num_mini_batches=32,
              num_learning_epochs=4,
              sampler='random',
              log_dir=log_dir,
              apply_reset=False)

    # ppo.load(path="run/model_0.pt")   # in case of resume=True
    ppo.run(num_learning_iterations=num_learning_iter, log_interval=50)


if __name__ == "__main__":
    print("My RL Project!")
    env_list = get_pybulletgym_env_list()
    env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
    # render_test(env_name, 10000)
    online_train(env_name=env_name, num_learning_iter=10, visualize=False)

    # env_name = "hopper-bullet-mixed-v0"
    # d3rlpy_dataset_check(env_name)
    # train_d3rlpy(env_name)


