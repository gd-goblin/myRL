import torch
import gym
import pybulletgym
from utils.env_parse import get_pybulletgym_env_list

import d3rlpy
from d3rlpy.algos import CQL

from task.task_desc import HopperBullet

from algo.ppo import PPO, ActorCritic


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def render_test(env_name, n_frames):
    env = gym.make(env_name)
    env.render()
    env.reset()
    for i in range(n_frames):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
    env.close()


def d3rlpy_dataset_check(env_name):
    dataset, env = d3rlpy.datasets.get_pybullet(env_name)
    print("dataset info::::")
    print("observations: ", dataset.observations.shape)
    print("actions: ", dataset.actions.shape)
    print("rewards: ", dataset.rewards.shape)
    print("terminals: ", dataset.terminals.shape)
    print("episodes: ", len(dataset.episodes))


def offline_train(env_name):
    dataset, env = d3rlpy.datasets.get_pybullet(env_name)

    cql = CQL(use_gpu=False)
    cql.fit(dataset,
            n_epochs=100,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer}
            )


def online_train(env_name, n_frames, visualize=False):
    env = HopperBullet(device)

    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=256,
              num_mini_batches=32,
              num_learning_epochs=8)

    ppo.run(num_learning_iterations=100, log_interval=50)

    return

    env.render() if visualize else None
    env.reset()
    for i in range(n_frames):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
    env.close()


if __name__ == "__main__":
    print("My RL Project!")
    env_list = get_pybulletgym_env_list()
    env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
    # render_test(env_name, 10000)
    online_train(env_name=env_name, n_frames=1000, visualize=False)

    # env_name = "hopper-bullet-mixed-v0"
    # d3rlpy_dataset_check(env_name)
    # train_d3rlpy(env_name)


