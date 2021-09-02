
import os
import gym
import pybulletgym
import d3rlpy

from d3rlpy.algos import CQL
import yaml


def get_pybulletgym_env_list():
    folder = 'config'
    file = 'env_list.yaml'
    path = os.path.join(folder, file)
    with open(path) as f:
        try:
            env_list = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    print("env list: ", env_list)
    return env_list


def render_test(env_name, n_frames):
    env = gym.make(env_name)
    env.render()
    env.reset()
    for i in range(n_frames):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
    env.close()


def dataset_check(env_name):
    dataset, env = d3rlpy.datasets.get_pybullet(env_name)
    print("dataset info::::")
    print("observations: ", dataset.observations.shape)
    print("actions: ", dataset.actions.shape)
    print("rewards: ", dataset.rewards.shape)
    print("terminals: ", dataset.terminals.shape)
    print("episodes: ", len(dataset.episodes))


def train_d3rlpy(env_name):
    dataset, env = d3rlpy.datasets.get_pybullet(env_name)

    cql = CQL(use_gpu=False)
    cql.fit(dataset,
            n_epochs=100,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer}
            )


if __name__ == "__main__":
    print("My RL Project!")
    env_list = get_pybulletgym_env_list()

    env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
    render_test(env_name, 10000)

    # env_name = "hopper-bullet-mixed-v0"
    # dataset_check(env_name)
    # train_d3rlpy(env_name)


