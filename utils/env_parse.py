import os
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