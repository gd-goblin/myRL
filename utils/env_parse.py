import os
import yaml
import d3rlpy


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
