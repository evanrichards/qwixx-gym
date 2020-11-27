import sys
from dataclasses import dataclass
from itertools import chain
from typing import List

from gym import spaces, Env
import numpy as np

RED = 0
YELLOW = 1
GREEN = 2
BLUE = 3

DIE_ROLLS = range(1, 7)
COLORS_MAX = {RED: 12, YELLOW: 12, GREEN: 2, BLUE: 2}
WHITE_ACTION_COLOR = [None, RED, YELLOW, GREEN, BLUE]
COLOR_ACTION = [None,
                ("white1", RED), ("white1", YELLOW),
                ("white1", GREEN), ("white1", BLUE),
                ("white2", RED), ("white2", YELLOW),
                ("white2", GREEN), ("white2", BLUE),
                ]
SCORE = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78]


def less_than(left, right):
    return left < right


def greater_than(left, right):
    return left > right


COMPARE_FUNCTION = {
    RED: greater_than, YELLOW: greater_than, GREEN: less_than, BLUE: less_than,
}
INVALID_MOVE_REWARD = -1
VALID_MOVE_REWARD = 0.01


class ObservationState(object):
    state = np.zeros(286, dtype=np.float32)
    bot_turn = state[0:1]
    players_turn = state[1:6]
    white1_roll = state[6:12]
    white2_roll = state[12:18]
    red_closed = state[18:19]
    red_roll = state[19:25]
    yellow_close = state[25:26]
    yellow_roll = state[26:32]
    green_close = state[32:33]
    green_roll = state[33:39]
    blue_close = state[39:40]
    blue_roll = state[40:46]

    player1_red = state[46:57]
    player1_yellow = state[57:68]
    player1_green = state[68:79]
    player1_blue = state[79:90]
    player1_strike = state[90:94]
    dice_set = [white1_roll, white2_roll, red_roll, yellow_roll, green_roll, blue_roll]
    player1_colors = [player1_red, player1_yellow, player1_green, player1_blue]
    player_colors = [player1_colors]
    player_strikes = [player1_strike]
    color_closed = [red_closed, yellow_close, green_close, blue_close]
    def __init__(self):
        self.reset()

    def reset(self):
        self.state[:] = 0

    def set_player(self, player_no, is_bot=False):
        self.bot_turn[0] = 1 if is_bot else 0
        self.players_turn[0:6] = 0
        self.players_turn[player_no] = 1

    def get_nums_for_color(self, color, player=0):
        if player is None:
            player = self.player_turn()
        counts = self.player_colors[player][color]
        l = list(range(2, 13)) if COMPARE_FUNCTION[color] == greater_than else list(
            range(12, 1, -1))
        return np.take(l,
                       np.nonzero(counts),
                       axis=0)[0]

    def is_closed(self, color):
        return self.color_closed[color][0] == 1

    def get_num_strikes(self, player=0):
        if player is None:
            player = self.player_turn()
        return int(sum(self.player_strikes[player]))


    def add_strike(self, player=0):
        if player is None:
            player = self.player_turn()
        self.player_strikes[player][self.get_num_strikes(player)] = 1

    def set_num_for_color(self, color, number, player=0):
        if player is None:
            player = self.player_turn()
        offset = ((number - 2) if COMPARE_FUNCTION[color] == greater_than else (12 - number))
        self.player_colors[player][color][offset] = 1
        if number == COLORS_MAX[color]:
            self.color_closed[color][0] = 1
            self._set_die(color + 2, 0)

    def set_dice(self, white1, white2, red, yellow, green, blue):
        for i, num in enumerate([white1, white2, red, yellow, green, blue]):
            if num != 0:
                self._set_die(i, num)

    def player_turn(self):
        return np.argmax(self.players_turn)

    def _set_die(self, offset, number):
        self.dice_set[offset][:] = 0
        if not number == 0:
            self.dice_set[offset][number - 1] = 1

    def get_dice_values(self):
        dice = []
        for i in range(0, 6):
            dice.append(np.argmax(self.dice_set[i]) + 1)
        return dice

    def dev_print(self):
        print("bots roll", bool(self.bot_turn[0]))
        print("player turn", "none" if not any(
            self.players_turn) else self.player_turn())
        dice = self.get_dice_values()
        print("dice", dice)
        print("white1", "not rolled" if not any(self.dice_set[0]) else dice[0])
        print("white2", "not rolled" if not any(self.dice_set[1]) else dice[1])
        print("red", ("closed" if self.color_closed[0][0] == 1 else (
            "not rolled" if dice[2] == 0 else dice[2])))
        print("yellow", ("closed" if self.color_closed[1][0] == 1 else (
            "not rolled" if dice[3] == 0 else dice[3])))
        print("green", ("closed" if self.color_closed[2][0] == 1 else (
            "not rolled" if dice[4] == 0 else dice[4])))
        print("blue", ("closed" if self.color_closed[3][0] == 1 else (
            "not rolled" if dice[5] == 0 else dice[5])))
        for i in range(0, 1):
            for j, color in enumerate(['red', 'yellow', 'green', 'blue']):
                print("player", i + 1, color, self.get_nums_for_color(j, i))
            print("player", i + 1, "strikes", self.get_num_strikes(i))


class QwixxNormalized(Env):
    """
    Action Space:
    Whites:
    - No take
    - Take as red
    - Take as yellow
    - Take as blue
    - Take as green
    Colors :
    - Take None
    - Take white 1 red
    - Take white 1 yellow
    - Take white 1 blue
    - Take white 1 green
    - Take white 2 red
    - Take white 2 yellow
    - Take white 2 blue
    - Take white 2 green
    """
    action_space = spaces.Discrete(45)
    observation_space = spaces.MultiBinary(286)
    env = object
    obs: ObservationState
    num_turns: int

    def __init__(self, num_players=5, bot_player=0,
                 valid_move_reward=VALID_MOVE_REWARD,
                 invalid_move_reward=INVALID_MOVE_REWARD):
        self.bot_player = bot_player
        self.num_players = num_players
        self.num_turns = 0
        self.invalid_move_reward = invalid_move_reward
        self.valid_move_reward = valid_move_reward
        self.obs = ObservationState()
        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        white_action, color_action = action % 5, action // 5
        if not self._is_bots_roll() and color_action != 0:
            # non-roller players can't take the color die
            return (self.obs.state, self.invalid_move_reward, True,
                    {"error": "took color, did not roll"})
        # If it chooses not to take white or color it adds a strike
        if self._is_bots_roll() and white_action == 0 and color_action == 0:
            self.obs.add_strike()
            self._advance_player()
            self._roll_dice()
            score = self._calculate_score()
            return (self.obs.state, score if self._is_done() else self.valid_move_reward,
                    self._is_done(), {"action": "took strike", "scores": score})
        # take white action
        if white_action != 0:
            err = self.process_white_action(white_action)
            if err is not None:
                return (self.obs.state, self.invalid_move_reward, True, err)
        # take color action
        if color_action != 0:
            err = self.process_color_action(color_action)
            if err is not None:
                return (self.obs.state, self.invalid_move_reward, True, err)
        self._advance_player()
        self._roll_dice()
        score = self._calculate_score()
        return (self.obs.state, score if self._is_done() else self.valid_move_reward,
                self._is_done(), {"scores": score})

    def process_color_action(self, color_action):
        white_die, color = COLOR_ACTION[color_action]
        nums = self.obs.get_nums_for_color(color)
        latest = nums[-1] if len(nums) > 0 else None
        cdv = self._color_dice_value(white_die, color)
        # Checks validity of move
        if latest is not None and not COMPARE_FUNCTION[color](cdv, latest):
            return {"error": "took color die invalid", "latest": latest,
                    "value": cdv, "color": color, "white": white_die,
                    "scores": self._calculate_score()}

        if cdv == COLORS_MAX[color] and not self._can_lock_color(color):
            return {"error": "tried to lock without having 5",
                    "latest": latest, "color": color,
                    "scores": self._calculate_score()}
        self.obs.set_num_for_color(color, cdv)

    def process_white_action(self, white_action):
        white_color = WHITE_ACTION_COLOR[white_action]
        nums = self.obs.get_nums_for_color(white_color)
        latest = nums[-1] if len(nums) > 0 else None
        wdv = self._white_dice_value()
        # Checks validity of move
        if latest is not None and not COMPARE_FUNCTION[white_color](wdv, latest):
            return {"error": "took white die invalid", "latest": latest,
                    "value": wdv, "color": white_color,
                    "scores": self._calculate_score()}
        if wdv == COLORS_MAX[white_color] and not self._can_lock_color(white_color):
            return {"error": "tried to lock without having 5",
                    "latest": latest, "color": white_color,
                    "scores": self._calculate_score()}
        self.obs.set_num_for_color(white_color, wdv)

    def reset(self):
        """Resets internal state to beginning of game and starts a new game"""
        self.current_player = 0
        self.obs.reset()
        self._roll_dice()
        self.num_turns = 0
        self.obs.set_player(0, is_bot=self.bot_player == 0)
        return self.obs.state

    def render(self, mode='human'):
        print("turn number ", self.num_turns)
        self.obs.dev_print()

    def _calculate_score(self):
        scores = []
        for color in COLORS_MAX.keys():
            nums = self.obs.get_nums_for_color(color)
            color_score = SCORE[len(nums)]
            if len(nums) > 0 and nums[-1] == COLORS_MAX[color]:
                color_score += 1
            scores.append(color_score)
        scores.append(self.obs.get_num_strikes() * -5)
        return sum(scores)

    def _can_lock_color(self, color):
        return len(self.obs.get_nums_for_color(color)) == 5

    def _is_done(self):
        if self.num_turns > 50:
            return True
        if self.obs.get_num_strikes() >= 4:
            return True
        self.locked_colors = []
        for color in COLORS_MAX.keys():
            if self.obs.is_closed(color):
                self.locked_colors.append(color)
        return len(self.locked_colors) >= 2

    def _roll_dice(self):
        self.num_turns += 1

        white1 = np.random.choice(DIE_ROLLS)
        white2 = np.random.choice(DIE_ROLLS)
        yellow, red, blue, green = np.zeros(4)
        if not self.obs.is_closed(YELLOW):
            yellow = np.random.choice(DIE_ROLLS)
        if not self.obs.is_closed(RED):
            red = np.random.choice(DIE_ROLLS)
        if not self.obs.is_closed(BLUE):
            blue = np.random.choice(DIE_ROLLS)
        if not self.obs.is_closed(GREEN):
            green = np.random.choice(DIE_ROLLS)
        self.obs.set_dice(white1, white2, red, yellow, green, blue)

    def _white_dice_value(self):
        dice = self.obs.get_dice_values()
        return sum(dice[0:2])

    def _color_dice_value(self, white_die, color):
        dice = self.obs.get_dice_values()
        return (dice[0] if white_die == "white1" else dice[1]) + dice[2 + color]

    def _is_bots_roll(self):
        return self.bot_player == self.obs.player_turn()

    def _advance_player(self):
        next_player = (self.obs.player_turn() + 1) % self.num_players
        self.obs.set_player(next_player, self.bot_player == next_player)
