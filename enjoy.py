import os
import time
import gym

import torch
from utils.env_parse import get_pybulletgym_env_list
from algo.ppo import PPO, ActorCritic
from task.task_desc import WrapperVecEnv


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def enjoy(env_name, n_frames):
    env = WrapperVecEnv(env_name=env_name, num_envs=1, device=device, normalized_env=False)

    init_noise_std = 1.0
    model_cfg = None
    model = ActorCritic(env.observation_space.shape, None, env.action_space.shape,
                        init_noise_std, model_cfg, asymmetric=False)
    try:
        # always loads the latest model if exists
        log_dir = os.path.join('run', env_name)
        matching = [s for s in os.listdir(log_dir) if "model_" in s]
        load_path = os.path.join(log_dir, matching[-1])

        model.load_state_dict(torch.load(load_path))
        model.eval()

        stat = [s for s in os.listdir(log_dir) if "vec_normalize_" in s]
        stat.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if stat[-1]:
            stat_path = os.path.join(log_dir, stat[-1])
            env.load(stat_path)

    except IndexError:
        print("No pre-trained model found")

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
        time.sleep(0.001)
    env.close()


if __name__ == "__main__":
    print("enjoy!")
    env_list = get_pybulletgym_env_list()
    env_name = env_list['PYBULLET_GYM_ENV_LIST'][4]
    enjoy(env_name, 10000)
