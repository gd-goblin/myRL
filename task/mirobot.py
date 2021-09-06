from assets.robots.robot_bases import URDFBasedRobot
import numpy as np
from utils.utils import *


class Mirobot(URDFBasedRobot):
    TARG_LIMIT = deg2rad(15)

    def __init__(self):
        URDFBasedRobot.__init__(self, model_urdf='mirobot_description\mirobot.urdf', robot_name='base_link', action_dim=8, obs_dim=6)
        print("Pybullet-Mirobot class description")

    def robot_specific_reset(self, physicsClient):
        print("jdict: ", self.jdict)
        print("parts: ", self.parts)
        self.jdict["joint1"].reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["joint2"].reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["joint3"].reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["joint4"].reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["joint5"].reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["joint6"].reset_current_position(np.random.uniform(low=-self.TARG_LIMIT, high=self.TARG_LIMIT), 0)
        self.jdict["right_finger_joint"].reset_current_position(0.017, 0)
        self.jdict["left_finger_joint"].reset_current_position(0.017, 0)

        self.hand = self.parts["mirobot_hand"]
        self.left_finger = self.parts["left_finger"]
        self.right_finger = self.parts["right_finger"]

        print("hand rot: ", self.hand.get_pose())

        hand_pose = self.hand.get_pose()
        lfinger_pose = self.left_finger.get_pose()
        rfinger_pose = self.right_finger.get_pose()
        self.draw_coordinate(origin=hand_pose[:3], quat=hand_pose[3:])
        self.draw_coordinate(origin=lfinger_pose[:3], quat=lfinger_pose[3:])
        self.draw_coordinate(origin=rfinger_pose[:3], quat=rfinger_pose[3:])

        # self.fingertip = self.parts["fingertip"]
        # self.target = self.parts["target"]
        # self.central_joint = self.jdict["joint0"]
        # self.elbow_joint = self.jdict["joint1"]
        # self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        # self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

    def apply_action(self, a):
        pass

    def calc_state(self):
        pass

    def calc_potential(self):
        pass

    def draw_coordinate(self, origin, quat, lenLine=0.1):
        mat = np.array(quat_to_mat(quat))
        px = origin + lenLine * mat[:, 0]
        py = origin + lenLine * mat[:, 1]
        pz = origin + lenLine * mat[:, 2]
        self._p.addUserDebugLine(origin, px, lineColorRGB=[1, 0, 0], lineWidth=2.0, lifeTime=0)
        self._p.addUserDebugLine(origin, py, lineColorRGB=[0, 1, 0], lineWidth=2.0, lifeTime=0)
        self._p.addUserDebugLine(origin, pz, lineColorRGB=[0, 0, 1], lineWidth=2.0, lifeTime=0)
