import os
import gym

import torch
from utils.env_parse import get_pybulletgym_env_list
from algo.ppo import PPO, ActorCritic
from task.task_desc import WrapperVecEnv


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def enjoy(env_name, n_frames):
    env = WrapperVecEnv(env_name=env_name, num_envs=1, device=device)

    # policy load
    path = "run"
    file_name = "model_350.pt"
    load_path = os.path.join(path, env_name, file_name)

    init_noise_std = 1.0
    model_cfg = None
    model = ActorCritic(env.observation_space.shape, None, env.action_space.shape,
                        init_noise_std, model_cfg, asymmetric=False)
    model.load_state_dict(torch.load(load_path))
    model.eval()

    env.render()
    obs = env.reset()
    frame = 0
    for i in range(n_frames):
        action = model.act_inference(obs)
        obs, reward, done, info = env.step(action.detach())

        frame += 1
        if done or frame > 300:
            obs = env.reset()
            print("stpes: {}, done: {}".format(i, done))
            frame = 0
    env.close()


if __name__ == "__main__":
    print("enjoy!")
    env_list = get_pybulletgym_env_list()
    env_name = env_list['PYBULLET_GYM_ENV_LIST'][8]
    enjoy(env_name, 10000)
