from qwixx_gym.envs import QwixxOneHotEnv

class QwixxSimple(QwixxOneHotEnv):
    def __init__(self, num_players=3, bot_player=0, invalid_move_reward=-1, default_return=0):
        super(QwixxSimple, self).__init__(num_players, bot_player, invalid_move_reward)
        self.default_return = default_return

    def calculate_skip_reward(self):
        if self._is_done():
            return self._calculate_score()
        return 0

    def calculate_reward(self, changed_values, current_score):
        if self._is_done():
            return self._calculate_score()
        return self.default_return
