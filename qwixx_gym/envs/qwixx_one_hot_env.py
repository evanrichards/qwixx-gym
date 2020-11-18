import sys
from dataclasses import dataclass
from itertools import chain
from typing import List

from gym import spaces, Env
import numpy as np


@dataclass
class ColorCount:
    red: int
    yellow: int
    green: int
    blue: int


@dataclass
class PlayerProgress:
    def __init__(self):
        self.counts = ColorCount(0, 0, 0, 0)
        self.latest_num = ColorCount(1, 1, 13, 13)
        self.strikes = 0


@dataclass
class Dice:
    white1: int = 0
    white2: int = 0
    red: int = 0
    yellow: int = 0
    green: int = 0
    blue: int = 0


DIE_ROLLS = range(1, 7)
COLORS_MAX = {"red": 12, "yellow": 12, "green": 2, "blue": 2}
WHITE_ACTION_COLOR = [None, "red", "yellow", "blue", "green"]
COLOR_ACTION = [None,
                ("white1", "red"), ("white1", "yellow"), ("white1", "blue"), ("white1", "green"),
                ("white2", "red"), ("white2", "yellow"), ("white2", "blue"), ("white2", "green"),
                ]
SCORE = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78]
SKIP_WEIGHT = [None, None, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]


def less_than(left, right):
    return left < right


def greater_than(left, right):
    return left > right


COMPARE_FUNCTION = {
    "red": greater_than, "yellow": greater_than, "green": less_than, "blue": less_than,
}
INVALID_MOVE_REWARD = 0
WIN_REWARD = 1000
LOSE_REWARD = 0
SKIP_BIAS = -2

class QwixxOneHotEnv(Env):
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
    action_space = spaces.MultiDiscrete([5, 9])
    action_space.n = 45
    dice: Dice
    previous_dice: Dice
    current_player: int
    progress: PlayerProgress
    env = object
    score: int

    def __init__(self, num_players=3, bot_player=0):
        self.bot_player = bot_player
        self.num_players = num_players
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1, 48), dtype='float32')
        self.current_player = 0
        self.num_turns = 0
        self.last_actions = []
        self.last_reward = None
        self.invalid_move_reward = INVALID_MOVE_REWARD
        self.win_reward = WIN_REWARD
        self.lose_reward = LOSE_REWARD
        self.skip_bias = SKIP_BIAS

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
        if not self.action_space.contains(action):
            # noinspection PyTypeChecker
            action = np.array([action % 5, action // 5])
        changed_values = []
        current_score = self._calculate_score()
        next_player = (self.current_player + 1) % self.num_players
        white_action, color_action = action
        self.last_actions = (WHITE_ACTION_COLOR[white_action], COLOR_ACTION[color_action])
        if not self._is_bots_roll() and color_action != 0:
            # non-roller players can't take the color die
            self.current_player = next_player
            return self._serialize_state(), self.invalid_move_reward, True, {}
            # {"error": "took color, did not roll", "scores": scores}
        # If it chooses not to take white or color it adds a strike
        if self._is_bots_roll() and white_action == 0 and color_action == 0:
            self.progress.strikes += 1
            self.current_player = next_player
            self._roll_dice()
            return (self._serialize_state(), self.calculate_skip_reward(),
                    self._is_done(), {"score": self._calculate_score()})
            # {"action": "took strike", "scores": scores}
        self.current_player = next_player
        # take white action
        if white_action != 0:
            white_color = WHITE_ACTION_COLOR[white_action]
            latest = self.progress.latest_num.__dict__[white_color]
            wdv = self._white_dice_value()
            # Checks validity of move
            if not COMPARE_FUNCTION[white_color](wdv, latest):
                return self._serialize_state(), self.invalid_move_reward, True, {"score": self._calculate_score()}
                # {"error": "took white die invalid", "latest": latest,
                #  "value": wdv, "color": white_color, "scores": scores})
            if wdv == COLORS_MAX[white_color] and not self._can_lock_color(white_color):
                return self._serialize_state(), self.invalid_move_reward, True, {"score": self._calculate_score()}
                # {"error": "tried to lock without having 5",
                #  "latest": self.progress[self.current_player].counts.__dict__[white_color],
                #  "color": white_color, "scores": scores})

            self.progress.counts.__dict__[white_color] += 1
            self.progress.latest_num.__dict__[white_color] = wdv
            changed_values.append((latest, wdv))

        # take color action
        if color_action != 0:
            white_die, color = COLOR_ACTION[color_action]
            latest = self.progress.latest_num.__dict__[color]
            cdv = self._color_dice_value(white_die, color)
            # Checks validity of move
            if not COMPARE_FUNCTION[color](cdv, latest):
                return self._serialize_state(), self.invalid_move_reward, True, {"score": self._calculate_score()}
                # {"error": "took color die invalid", "latest": latest,
                #  "value": cdv, "color": color, "white": white_die, "scores": scores})

            if cdv == COLORS_MAX[color] and not self._can_lock_color(color):
                self.current_player = next_player
                return self._serialize_state(), self.invalid_move_reward, True, {"score": self._calculate_score()}
                # {"error": "tried to lock without having 5",
                #  "latest": self.progress[self.current_player].counts.__dict__[color],
                #  "color": color, "scores": scores})

            self.progress.counts.__dict__[color] += 1
            self.progress.latest_num.__dict__[color] = cdv
            changed_values.append((latest, cdv))
        self.current_player = next_player
        reward = self.calculate_reward(changed_values, current_score)
        self._roll_dice()
        return (self._serialize_state(), reward,
                self._is_done(), {"score": self._calculate_score()})  # {"scores": scores}

    def calculate_skip_reward(self):
        wdv = self._white_dice_value()
        skipped = []
        for color, latest in self.progress.latest_num.__dict__.items():
            cmv = COLORS_MAX[color]
            # check if we have passed the white value
            if cmv >= latest and latest >= wdv:
                continue
            if wdv >= latest and latest >= cmv:
                continue
            mi = min([latest, wdv])
            ma = max([latest, wdv])
            if mi < ma:
                skipped.append(len(SKIP_WEIGHT[mi + 1: ma]))
        if skipped:
            r = min(skipped)
        else:
            r = 10
        self.last_reward = r
        if not self._is_bots_roll():
            return self.last_reward
        # if bot rolled we should see is colors were skipped too
        skipped = []
        for color, latest in self.progress.latest_num.__dict__.items():
            min_skip_distance = self.get_skipped_values(color, latest)
            if min_skip_distance is not None:
                skipped.append(min_skip_distance)
        if skipped:
            r = min(skipped)
        else:
            r = 10
        self.last_reward = min([self.last_reward, r])
        return self.last_reward - self.skip_bias

    def get_skipped_values(self, color, latest):
        cmv = COLORS_MAX[color]
        cval1 = self._color_dice_value('white1', color)
        cval2 = self._color_dice_value('white2', color)
        if cmv == latest:
            return None
        if cmv > latest:
            vals = []
            if cval1 - latest > 0:
                vals.append(cval1)
            if cval2 - latest > 0:
                vals.append(cval2)
            if not vals:
                return None
            return len(SKIP_WEIGHT[latest + 1: min(vals)])
        vals = []
        if latest - cval1 > 0:
            vals.append(cval1)
        if latest - cval2 > 0:
            vals.append(cval2)
        if not vals:
            return None
        return len(SKIP_WEIGHT[max(vals): latest - 1])

    def calculate_reward(self, changed_values, current_score):
        if not changed_values:
            return self.calculate_skip_reward()
        skipped = []
        for before, after in changed_values:
            if after - before >= 0:
                skipped.append(sum(SKIP_WEIGHT[before + 1: after]))
            else:
                skipped.append(sum(SKIP_WEIGHT[after + 1: before]))
        divisor = sum(skipped) + 1
        r = (self._calculate_score() - current_score) / divisor * 100
        self.last_reward = r
        return r

    def reset(self):
        """Resets internal state to beginning of game and starts a new game"""
        self.dice = Dice()
        self.last_actions = []
        self.current_player = 0
        self.progress = PlayerProgress()
        self._roll_dice()
        self.num_turns = 0
        self.previous_dice = None
        self.last_reward = None
        self.score = 0
        return self._serialize_state()

    def render(self, mode='human'):
        if self.previous_dice is not None:
            print("Previous dice: ", self.previous_dice)
        if len(self.last_actions) == 2:
            print("Last action: Whites: {}, Colors: {}".format(self.last_actions[0], self.last_actions[1]))
            print("Last reward: ", self.last_reward)
        print("\tLatest:", self.progress.latest_num)
        print("\tCounts:", self.progress.counts)
        print("\tStrikes:", self.progress.strikes)
        print("\tScore:", self._calculate_score())
        print("Num turns:", self.num_turns)
        print("Rolling player:", self.current_player)
        print("New dice:", self.dice)
        print("")

    def _calculate_score(self):
        scores = []
        for color, count in self.progress.counts.__dict__.items():
            color_score = SCORE[count]
            if self.progress.latest_num.__dict__[color] == COLORS_MAX[color]:
                color_score += 1
            scores.append(color_score)
        scores.append(self.progress.strikes * -5)
        self.score = sum(scores)
        return self.score

    def _color_is_locked(self, color):
        max_value = COLORS_MAX[color]
        return self.progress.latest_num.__dict__[color] == max_value

    def _can_lock_color(self, color):
        return self.progress.counts.__dict__[color] == 5

    def _is_done(self):
        if self.num_turns > 50:
            return True
        self.locked_colors = []
        for color, max_value in COLORS_MAX.items():
            if self.progress.strikes == 4:
                return True
            if self.progress.latest_num.__dict__[color] == max_value:
                self.locked_colors.append(color)
        return len(self.locked_colors) >= 2

    def _roll_dice(self):
        self.num_turns += 1
        self.previous_dice = Dice(**self.dice.__dict__)
        self.dice.white1 = np.random.choice(DIE_ROLLS)
        self.dice.white2 = np.random.choice(DIE_ROLLS)

        if not self._color_is_locked("yellow"):
            self.dice.yellow = np.random.choice(DIE_ROLLS)
        if not self._color_is_locked("red"):
            self.dice.red = np.random.choice(DIE_ROLLS)
        if not self._color_is_locked("blue"):
            self.dice.blue = np.random.choice(DIE_ROLLS)
        if not self._color_is_locked("green"):
            self.dice.green = np.random.choice(DIE_ROLLS)

    def _white_dice_value(self):
        return self.dice.white1 + self.dice.white2

    def _color_dice_value(self, white_die, color):
        dice = self.dice.__dict__
        return dice[white_die] + dice[color]

    def _is_bots_roll(self):
        return self.bot_player == self.current_player

    def _serialize_state(self):
        y = self._is_bots_roll()
        a = float(y)
        return np.array([
            a,
            self.current_player,
            self.bot_player,
            *self._serialize_dice(),
            *self._serialize_player(),
        ], dtype=np.float)

    def _serialize_player(self) -> List[float]:
        """
        Serializes a players board. There are only 11 possible crosses you can do,
        this excludes the lock and the number 1. Since we store the latest numbers
        as their actual number, to get it to a 0.0-1.0 scale, we subtract one from
        red and yellow (can't roll a 1) and subtract 2 from green and blue (can't
        roll a 1 and not counting the lock)
        """
        return [
            float(self.progress.counts.red),
            float(self.progress.counts.yellow),
            float(self.progress.counts.green),
            float(self.progress.counts.blue),
            float((self.progress.latest_num.red - 1) / 11.0),
            float((self.progress.latest_num.yellow - 1) / 11.0),
            1 - float((self.progress.latest_num.green - 2) / 11.0),
            1 - float((self.progress.latest_num.blue - 2) / 11.0),
            float(self.progress.strikes / 4.0),
        ]

    def _serialize_dice(self):
        return [
            *[1. if x == self.dice.white1 - 1 else 0.0 for x in range(6)],
            *[1. if x == self.dice.white2 - 1 else 0.0 for x in range(6)],
            *[1. if x == self.dice.red - 1 else 0.0 for x in range(6)],
            *[1. if x == self.dice.yellow - 1 else 0.0 for x in range(6)],
            *[1. if x == self.dice.green - 1 else 0.0 for x in range(6)],
            *[1. if x == self.dice.blue - 1 else 0.0 for x in range(6)],
        ]
