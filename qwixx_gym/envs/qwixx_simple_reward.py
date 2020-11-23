from qwixx_gym.envs import QwixxOneHotEnv

class QwixxSimple(QwixxOneHotEnv):
    def __init__(self, default_return=0, **kwargs):
        super(QwixxSimple, self).__init__(**kwargs)
        self.default_return = default_return

    def calculate_skip_reward(self):
        if self._is_done():
            return self._calculate_score()
        return 0

    def calculate_reward(self, changed_values, current_score):
        if self._is_done():
            return self._calculate_score()
        return self.default_return
