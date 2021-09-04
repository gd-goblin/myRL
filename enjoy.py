import os
import gym

import torch
from utils.env_parse import get_pybulletgym_env_list
from algo.ppo import PPO, ActorCritic
from task.task_desc import HopperBullet


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def enjoy(env_name, n_frames):
    env = HopperBullet(device=device)

    # policy load
    path = "run"
    file_name = "model_650.pt"
    load_path = os.path.join(path, file_name)

    init_noise_std = 1.0
    model_cfg = None
    model = ActorCritic(env.observation_space.shape, None, env.action_space.shape,
                        init_noise_std, model_cfg, asymmetric=False)
    model.load_state_dict(torch.load(load_path))
    model.eval()

    env.render()
    obs = env.reset()
    for i in range(n_frames):
        action = model.act_inference(obs)
        obs, reward, done, info = env.step(action.detach())

        if done:
            obs = env.reset()
            print("stpes: {}, done: {}".format(i, done))
    env.close()


if __name__ == "__main__":
    print("enjoy!")
    env_list = get_pybulletgym_env_list()
    env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
    enjoy(env_name, 10000)
