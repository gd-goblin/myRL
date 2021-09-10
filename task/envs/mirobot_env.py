from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
from task.mirobot import Mirobot
import pybullet_data
import numpy as np


class MirobotBulletEnv(BaseBulletEnv):
    def __init__(self, draw_debug_lines=False):
        self.env_name = 'MirobotPyBullet'
        self.robot = Mirobot(draw_debug_lines)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.807, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        # electricity_cost = (
        #         -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot))  # work torque*angular_velocity
        #         - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        # )
        # stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0

        # work: torque * angular_velocity
        # stall torque require some energy
        electricity_cost = (
            -0.1 * (np.abs(a[0] * self.robot.theta_dot1) +
                    np.abs(a[1] * self.robot.theta_dot2) +
                    np.abs(a[2] * self.robot.theta_dot3) +
                    np.abs(a[3] * self.robot.theta_dot4) +
                    np.abs(a[4] * self.robot.theta_dot5) +
                    np.abs(a[5] * self.robot.theta_dot6))
            - 0.01 * (np.abs(a[0]) + np.abs(a[1]) + np.abs(a[2]) + np.abs(a[3]) + np.abs(a[4]) + np.abs(a[5]))
        )
        self.rewards = [float(self.potential - potential_old), float(electricity_cost)]  # float(stuck_joint_cost)
        self.HUD(state, a, False)

        if self.robot.draw_debug_lines:
            finger_tip_pose = self.robot.get_finger_tip_pose()
            self.robot.draw_coordinate(origin=finger_tip_pose[:3], quat=finger_tip_pose[3:])

        return state, sum(self.rewards), False, {}

    def episode_reset(self):
        self.robot.episode_reset()

    def camera_adjust(self):
        x, y, z = self.robot.get_finger_tip_pose()[:3]
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
