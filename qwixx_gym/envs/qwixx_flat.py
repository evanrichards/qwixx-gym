from qwixx_gym.envs import QwixxSimple
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

class QwixxFlat(QwixxSimple):
    action_space = Discrete(45)
    def step(self, action):
        return super(QwixxFlat, self).step(np.array([action % 5, action // 5]))
