from qwixx_gym.envs import QwixxOneHotEnv

class QwixxSimple(QwixxOneHotEnv):
    def __init__(self, num_players=3, bot_player=0):
        super(QwixxSimple, self).__init__(num_players, bot_player)
        self.invalid_move_reward = 0
        self.win_reward = 0
        self.lose_reward = 0
        self.skip_bias = 0

    def calculate_skip_reward(self):
        return 0

    def calculate_reward(self):
        if self._is_done():
            return self._calculate_score()
        return 0
