
import gym
import pybulletgym
import d3rlpy

env_list = ["InvertedPendulumPyBulletEnv-v0",
            "InvertedDoublePendulumPyBulletEnv-v0",
            "InvertedPendulumSwingupPyBulletEnv-v0",
            "ReacherPyBulletEnv-v0",
            "Walker2DPyBulletEnv-v0",
            "HalfCheetahPyBulletEnv-v0",
            "AntPyBulletEnv-v0",
            "HopperPyBulletEnv-v0",
            "HumanoidPyBulletEnv-v0",
            "HumanoidFlagrunPyBulletEnv-v0",
            "HumanoidFlagrunHarderPyBulletEnv-v0",
            "AtlasPyBulletEnv-v0",
            "PusherPyBulletEnv-v0",
            "ThrowerPyBulletEnv-v0",
            "StrikerPyBulletEnv-v0",
            "InvertedPendulumMuJoCoEnv-v0",
            "InvertedDoublePendulumMuJoCoEnv-v0",
            "ReacherMuJoCoEnv-v0",
            "Walker2DMuJoCoEnv-v0",
            "HalfCheetahMuJoCoEnv-v0",
            "AntMuJoCoEnv-v0",
            "HopperMuJoCoEnv-v0",
            "HumanoidMuJoCoEnv-v0",
            "PusherMuJoCoEnv-v0",
            "ThrowerMuJoCoEnv-v0",
            "StrikerMuJoCoEnv-v0"]


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
    print("dataset info: ", dataset)


if __name__ == "__main__":
    print("My RL Project!")
    # env_name = env_list[0]
    # render_test(env_name, 1000)
    dataset_check("hopper-bullet-mixed-v0")

