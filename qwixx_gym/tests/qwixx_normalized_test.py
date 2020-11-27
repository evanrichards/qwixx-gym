import unittest

from qwixx_gym.envs.qwixx_normalized import ObservationState, RED, YELLOW, GREEN, BLUE
import numpy as np


class QwixxObservationState(unittest.TestCase):
    def test_reset(self):
        obs = ObservationState()
        obs.set_dice(1, 2, 3, 4, 5, 6)
        obs.set_player(0, True)
        self.assertEqual(0, obs.player_turn())
        self.assertEqual(1, obs.state[0])
        self.assertEqual(1, obs.players_turn[0], obs.players_turn)
        obs.reset()
        self.assertTrue(not any(obs.state))

    def test_get_dice(self):
        obs = ObservationState()
        obs.set_dice(1, 2, 3, 4, 5, 6)
        self.assertEqual([1, 2, 3, 4, 5, 6],
                         obs.get_dice_values(), obs.dev_print())
                         

    def test_set_color(self):
        obs = ObservationState()
        obs.set_num_for_color(BLUE, 2, 0)
        obs.set_num_for_color(BLUE, 3, 0)
        obs.set_num_for_color(BLUE, 4, 0)
        obs.set_num_for_color(BLUE, 11, 0)
        res = obs.get_nums_for_color(BLUE, 0)
        self.assertEqual(4, len(res), obs.dev_print())
        self.assertTrue(
            all(map(lambda a: a[0] == a[1], zip([11, 4, 3, 2], res))), res)
        self.assertTrue(obs.is_closed(BLUE), obs.dev_print())

    def test_strikes(self):
        obs = ObservationState()
        self.assertEqual(0, obs.get_num_strikes(0))
        obs.add_strike(0)
        obs.add_strike(0)
        obs.add_strike(0)
        self.assertEqual(3, obs.get_num_strikes(0))
