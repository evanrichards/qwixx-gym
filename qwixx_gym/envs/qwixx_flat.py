from qwixx_gym.envs import QwixxSimple
from gym.spaces import Discrete
import numpy as np

class QwixxFlat(QwixxSimple):
    action_space = Discrete(45)
    def step(self, action):
        if not super(QwixxFlat, self).action_space.contains(action):
            action = np.array([action % 5, action // 5])
        return super(QwixxFlat, self).step(action)
