import unittest
from qwixx_gym.envs.qwixx_one_hot_env import QwixxOneHotEnv, Dice, ColorCount


class QwixxOneHotEnvTest(unittest.TestCase):

    def setUp(self) -> None:
        self.env = QwixxOneHotEnv()

    def test_calculate_skip_reward_white_dice_not_roller(self):
        self.env.current_player = 1
        self.env.bot_player = 0
        for x in range(6):
            self.env.dice = Dice(x + 1, 1, 6, 6, 1, 1)
            reward = self.env.calculate_reward([], 0)
            self.assertEqual(x, reward, "count up should increase {}".format(x))
        self.env.progress.latest_num.green = 2
        self.env.progress.latest_num.blue = 2
        for x in range(6):
            self.env.dice = Dice(x + 1, 6, 6, 6, 1, 1)
            reward = self.env.calculate_reward([], 0)
            self.assertEqual(x + 5, reward, "worst case can take should be 10")

    def test_calculate_skip_reward_color_dice_roller(self):
        self.env.current_player = self.env.bot_player
        self.env.dice = Dice(1, 6, 1, 1, 1, 1)
        reward = self.env.calculate_reward([], 0)
        self.assertEqual(0, reward, "adjacent numbers should result in zero reward")
        self.env.dice = Dice(1, 1, 6, 6, 1, 1)
        self.env.progress.latest_num.green = 2
        self.env.progress.latest_num.blue = 2
        self.env.progress.latest_num.red = 2
        self.env.progress.latest_num.yellow = 2
        reward = self.env.calculate_reward([], 0)
        self.assertEqual(4, reward, "should skip 3, 4, 5, 6")

    def test_calculate_score_worst_case(self):
        self.env.current_player = self.env.bot_player
        self.env.dice = Dice(6, 6, 6, 6, 6, 6)
        self.env.progress.latest_num = ColorCount(12, 12, 2, 2)
        reward = self.env.calculate_reward([], 0)
        self.assertEqual(10, reward, "can't take any dice should be 10")
        self.env.dice = Dice(1, 1, 1, 1, 1, 1)
        self.env.progress.latest_num = ColorCount(12, 12, 2, 2)
        reward = self.env.calculate_reward([], 0)
        self.assertEqual(10, reward, "cant take any bc lockout should be 10")
        self.env.progress.latest_num = ColorCount(11, 11, 2, 2)
        reward = self.env.calculate_reward([], 0)
        self.assertEqual(10, reward, "can't take any dice should be 10")

    def test_failure(self):
        self.env.current_player = self.env.bot_player + 1
        self.env.dice = Dice(2, 2, 2, 4, 6, 6)
        self.env.progress.latest_num = ColorCount(1, 2, 10, 10)
        self.env.progress.counts = ColorCount(0, 3, 0, 0)

        reward = self.env.calculate_reward([(2, 4), (4, 5)], 3)
        self.assertEqual(3, reward)


if __name__ == '__main__':
    unittest.main()
