from assets.robots.robot_bases import URDFBasedRobot
import numpy as np
import os
import pybullet
import pybullet_data
from utils.utils import *


class Mirobot(URDFBasedRobot):
    TARG_LIMIT = deg2rad(15)

    def __init__(self, draw_debug_lines=False):
        URDFBasedRobot.__init__(self, model_urdf='mirobot.urdf', robot_name='mirobot_urdf', fixed_base=True, action_dim=7, obs_dim=16)
        print("Pybullet-Mirobot class description")

        self.draw_debug_lines = draw_debug_lines
        self.debug_line_id = [None, None, None]

    def reset(self, bullet_client):     # overriding
        self._p = bullet_client
        self.ordered_joints = []

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.parts, self.jdict, self.ordered_joints, self.robot_body = \
            self.addToScene(self._p,
                            self._p.loadURDF('plane.urdf',
                                             basePosition=[0, 0, 0],
                                             baseOrientation=[0, 0, 0, 1],
                                             useFixedBase=True))

        robot_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'robots', 'mirobot_description', self.model_urdf)
        print(robot_path)

        flags = pybullet.URDF_USE_SELF_COLLISION if self.self_collision else 0
        self.parts, self.jdict, self.ordered_joints, self.robot_body = \
            self.addToScene(self._p,
                            self._p.loadURDF(robot_path,
                                             basePosition=self.basePosition,
                                             baseOrientation=self.baseOrientation,
                                             useFixedBase=self.fixed_base,
                                             flags=flags))

        obs_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'objects', 'cube.urdf')
        print(obs_path)
        self.parts, self.jdict, self.ordered_joints, self.robot_body = \
            self.addToScene(self._p,
                            self._p.loadURDF(obs_path,
                                             basePosition=[0.2, 0, 0],
                                             baseOrientation=[0, 0, 0, 1],
                                             useFixedBase=False))

        self.robot_specific_reset(self._p)

        s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()
        return s

    def robot_specific_reset(self, physicsClient):
        print("jdict: ", self.jdict)
        print("parts: ", self.parts)
        self.j1 = self.jdict["joint1"]
        self.j2 = self.jdict["joint2"]
        self.j3 = self.jdict["joint3"]
        self.j4 = self.jdict["joint4"]
        self.j5 = self.jdict["joint5"]
        self.j6 = self.jdict["joint6"]
        self.jlfinger = self.jdict["left_finger_joint"]
        self.jrfinger = self.jdict["right_finger_joint"]
        self.grip = 0

        self.j1.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j2.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j3.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j4.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j5.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j6.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jlfinger.reset_current_position(0.017, 0)
        self.jrfinger.reset_current_position(0.017, 0)

        self.p_hand = self.parts["mirobot_hand"]
        self.p_left_finger = self.parts["left_finger"]
        self.p_right_finger = self.parts["right_finger"]
        self.p_cube = self.parts["cube"]

        quat = euler_to_quat(roll=0.0, pitch=0.0, yaw=np.random.uniform(low=-deg2rad(90), high=deg2rad(90)))
        pos = np.array([np.random.uniform(0.1, 0.3), np.random.uniform(-0.15, 0.15), 0.1])
        self.p_cube.reset_pose(position=pos, orientation=quat)

        if self.draw_debug_lines:
            finger_tip_pose = self.get_finger_tip_pose()
            self.draw_coordinate(origin=finger_tip_pose[:3], quat=finger_tip_pose[3:])

    def episode_reset(self):
        self.j1.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j2.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j3.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j4.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j5.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.j6.reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jlfinger.reset_current_position(0.017, 0)
        self.jrfinger.reset_current_position(0.017, 0)

        quat = euler_to_quat(roll=0.0, pitch=0.0, yaw=np.random.uniform(low=-deg2rad(90), high=deg2rad(90)))
        pos = np.array([np.random.uniform(0.1, 0.3), np.random.uniform(-0.15, 0.15), 0.1])
        self.p_cube.reset_pose(position=pos, orientation=quat)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        assert (a.shape == self.action_space.shape)
        self.j1.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.j2.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))
        self.j3.set_motor_torque(0.05 * float(np.clip(a[2], -1, +1)))
        self.j4.set_motor_torque(0.05 * float(np.clip(a[3], -1, +1)))
        self.j5.set_motor_torque(0.05 * float(np.clip(a[4], -1, +1)))
        self.j6.set_motor_torque(0.05 * float(np.clip(a[5], -1, +1)))
        self.grip = a[-1]
        grip_act = 0 if self.grip else 0.017
        self.jlfinger.set_position(grip_act)
        self.jrfinger.set_position(grip_act)

    def calc_state(self):
        # gripper pose & vel
        grip_pose = self.get_finger_tip_pose()
        grip_x, grip_y, grip_z, grip_qx, grip_qy, grip_qz, grip_qw = grip_pose
        hand_lin_vel, hand_rot_vel = self.p_hand.get_velocity()

        # cube pose & vel
        cube_pose = self.p_cube.get_pose()
        cube_x, cube_y, cube_z, cube_qx, cube_qy, cube_qz, cube_qw = cube_pose
        cube_lin_vel, cube_rot_vel = self.p_cube.get_velocity()

        to_cube_vec = cube_pose[:3] - grip_pose[:3]

        # joint angles & angular vel
        theta1, self.theta_dot1 = self.j1.current_relative_position()
        theta2, self.theta_dot2 = self.j2.current_relative_position()
        theta3, self.theta_dot3 = self.j3.current_relative_position()
        theta4, self.theta_dot4 = self.j4.current_relative_position()
        theta5, self.theta_dot5 = self.j5.current_relative_position()
        theta6, self.theta_dot6 = self.j6.current_relative_position()

        return np.array([
            to_cube_vec[0], to_cube_vec[1], to_cube_vec[2],
            theta1, self.theta_dot1,
            theta2, self.theta_dot2,
            theta3, self.theta_dot3,
            theta4, self.theta_dot4,
            theta5, self.theta_dot5,
            theta6, self.theta_dot6,
            self.grip
        ])

    def calc_potential(self):
        grip_pose = self.get_finger_tip_pose()
        cube_pose = self.p_cube.get_pose()
        to_cube_vec = cube_pose[:3] - grip_pose[:3]
        return -100 * np.linalg.norm(to_cube_vec)

    def get_finger_tip_pose(self):
        lfinger_pose = self.p_left_finger.get_pose()
        rfinger_pose = self.p_right_finger.get_pose()
        center = 0.5 * (lfinger_pose[:3] + rfinger_pose[:3])
        m = quat_to_mat(lfinger_pose[3:])
        center += m[:, 2] * 0.023    # z-offset
        return np.concatenate((center, lfinger_pose[3:]), axis=0)

    def draw_coordinate(self, origin, quat, lenLine=0.1):
        for id in self.debug_line_id:
            if id is not None:
                self._p.removeUserDebugItem(id)

        mat = quat_to_mat(quat)
        px = origin + lenLine * mat[:, 0]
        py = origin + lenLine * mat[:, 1]
        pz = origin + lenLine * mat[:, 2]
        self.debug_line_id[0] = self._p.addUserDebugLine(origin, px, lineColorRGB=[1, 0, 0], lineWidth=2.0, lifeTime=0)
        self.debug_line_id[1] = self._p.addUserDebugLine(origin, py, lineColorRGB=[0, 1, 0], lineWidth=2.0, lifeTime=0)
        self.debug_line_id[2] = self._p.addUserDebugLine(origin, pz, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
