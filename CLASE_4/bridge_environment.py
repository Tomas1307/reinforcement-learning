import numpy as np
from typing import Tuple, List


class BridgeEnvironment:
    def __init__(self):
        self.nrows = 3
        self.ncols = 7
        self.initial_state = (1, 0)
        self.current_state = self.initial_state

        self.board = [
            [' ', -100, -100, -100, -100, -100, ' '],
            ['S',  ' ',  ' ',  ' ',  ' ',  ' ', 100],
            [' ', -100, -100, -100, -100, -100, ' '],
        ]

        self.action_success_prob = 0.60
        self.clockwise_prob = 0.20
        self.counterclockwise_prob = 0.10
        self.stay_prob = 0.10

        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def _get_clockwise_action(self, action):
        clockwise_map = {'up': 'right', 'right': 'down', 'down': 'left', 'left': 'up'}
        return clockwise_map[action]

    def _get_counterclockwise_action(self, action):
        counterclockwise_map = {'up': 'left', 'left': 'down', 'down': 'right', 'right': 'up'}
        return counterclockwise_map[action]

    def _calculate_new_state(self, r, c, action):
        if action not in self.action_map:
            return (r, c)
        dr, dc = self.action_map[action]
        new_r, new_c = r + dr, c + dc

        if (new_r < 0 or new_r >= self.nrows or
            new_c < 0 or new_c >= self.ncols or
            self.board[new_r][new_c] == '#'):
            return (r, c)

        return (new_r, new_c)

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        r, c = state
        if isinstance(self.board[r][c], (int, float)):
            return ['exit']
        return ['up', 'down', 'left', 'right']

    def do_action(self, action):
        r, c = self.current_state

        if self.is_terminal():
            if action == 'exit':
                reward = self.board[r][c]
                return reward, self.current_state
            return 0, self.current_state

        clockwise = self._get_clockwise_action(action)
        counterclockwise = self._get_counterclockwise_action(action)

        rand = np.random.random()
        if rand < self.action_success_prob:
            executed = action
        elif rand < self.action_success_prob + self.clockwise_prob:
            executed = clockwise
        elif rand < self.action_success_prob + self.clockwise_prob + self.counterclockwise_prob:
            executed = counterclockwise
        else:
            executed = None

        if executed is None:
            new_state = (r, c)
        else:
            new_state = self._calculate_new_state(r, c, executed)

        reward = 0
        if isinstance(self.board[new_state[0]][new_state[1]], (int, float)):
            reward = self.board[new_state[0]][new_state[1]]

        self.current_state = new_state
        return reward, new_state

    def reset(self):
        self.current_state = self.initial_state

    def is_terminal(self):
        r, c = self.current_state
        return isinstance(self.board[r][c], (int, float))
