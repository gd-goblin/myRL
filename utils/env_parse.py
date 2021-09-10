import os
import yaml
import d3rlpy
import gym

from stable_baselines3.common.utils import set_random_seed


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_custom_env(env_class, rank=0, seed=0):

    def _init():
        env = env_class()
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def _make_custom_env(env_class, seed=0):    # for single custom env
    env = env_class()
    set_random_seed(seed)
    env.seed(seed)
    return env


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


def d3rlpy_dataset_check(env_name):
    dataset, env = d3rlpy.datasets.get_pybullet(env_name)
    print("dataset info::::")
    print("observations: ", dataset.observations.shape)
    print("actions: ", dataset.actions.shape)
    print("rewards: ", dataset.rewards.shape)
    print("terminals: ", dataset.terminals.shape)
    print("episodes: ", len(dataset.episodes))
