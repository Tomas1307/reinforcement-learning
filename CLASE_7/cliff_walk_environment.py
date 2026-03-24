import numpy as np
from loguru import logger


class CliffWalk:
    """
    Cliff-Walk environment based on GridWorld.
    6x12 grid with a cliff along the bottom row.

    - Start: bottom-left (5, 0)
    - Goal: bottom-right (5, 11)
    - Cliff: bottom row from (5,1) to (5,10) -> reward -100, reset to start
    - Step reward: -1
    - Deterministic actions
    """

    def __init__(self):
        self.nrows = 6
        self.ncols = 12
        self.initial_state = (5, 0)
        self.current_state = self.initial_state

        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        self.cliff = set()
        for c in range(1, 11):
            self.cliff.add((5, c))

        self.goal = (5, 11)

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        return self.actions

    def do_action(self, action):
        """Execute action deterministically. Returns (reward, new_state)."""
        r, c = self.current_state
        dr, dc = self.action_map[action]
        new_r, new_c = r + dr, c + dc

        if new_r < 0 or new_r >= self.nrows or new_c < 0 or new_c >= self.ncols:
            new_r, new_c = r, c

        new_state = (new_r, new_c)

        if new_state in self.cliff:
            self.current_state = self.initial_state
            return -100, self.initial_state

        self.current_state = new_state
        return -1, new_state

    def is_terminal(self):
        return self.current_state == self.goal

    def reset(self):
        self.current_state = self.initial_state

    def get_states(self):
        states = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                states.append((r, c))
        return states

    def get_reward(self, action, state, new_state):
        """Calculate reward for a transition."""
        if new_state in self.cliff:
            return -100
        return -1

    def print_grid(self, path=None):
        """Print the grid with optional path visualization."""
        for r in range(self.nrows):
            row_str = ""
            for c in range(self.ncols):
                if (r, c) == self.initial_state:
                    row_str += " S "
                elif (r, c) == self.goal:
                    row_str += " G "
                elif (r, c) in self.cliff:
                    row_str += " X "
                elif path and (r, c) in path:
                    row_str += " . "
                else:
                    row_str += " - "
            print(row_str)
        print()
