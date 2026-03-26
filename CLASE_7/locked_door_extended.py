import numpy as np
from loguru import logger


class LockedDoorExtended:
    """
    Extended Locked-Door environment for adaptability tests.

    State includes key position: (row, col, has_ball, has_key, door_open, key_row, key_col)
    This allows the agent to generalize across different key positions.
    """

    def __init__(self, nrows=4, ncols=9, wall_col=4, door_row=3,
                 agent_start=None, key_pos=None, ball_pos=None, goal_pos=None,
                 key_color='blue', key_positions=None, randomize_start=False):
        self.nrows = nrows
        self.ncols = ncols
        self.wall_col = wall_col
        self.door_row = door_row
        self.key_color = key_color
        self.door_color = 'blue'

        self.agent_start = agent_start or (2, 0)
        self.key_pos = key_pos or (1, 3)
        self.ball_pos = ball_pos or (3, 3)
        self.goal_pos = goal_pos or (0, 6)

        self.randomize_start = randomize_start
        self.key_positions = key_positions or [(r, c) for r in range(nrows) for c in range(wall_col)]
        self.start_positions = [(r, c) for r in range(nrows) for c in range(wall_col)]

        self.current_state = (
            self.agent_start[0], self.agent_start[1],
            False, False, False,
            self.key_pos[0], self.key_pos[1]
        )

        self.actions = ['up', 'down', 'left', 'right', 'pick_up', 'open_door']
        self.action_map = {
            'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)
        }

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        return self.actions

    def _is_wall(self, r, c, door_open):
        if r < 0 or r >= self.nrows or c < 0 or c >= self.ncols:
            return True
        if c == self.wall_col:
            if r == self.door_row:
                return not door_open
            return True
        return False

    def do_action(self, action):
        r, c, has_ball, has_key, door_open, kr, kc = self.current_state

        if action in self.action_map:
            dr, dc = self.action_map[action]
            new_r, new_c = r + dr, c + dc

            if self._is_wall(new_r, new_c, door_open):
                self.current_state = (r, c, has_ball, has_key, door_open, kr, kc)
                return -1, self.current_state

            self.current_state = (new_r, new_c, has_ball, has_key, door_open, kr, kc)
            if (new_r, new_c) == self.goal_pos:
                return 100, self.current_state
            return -1, self.current_state

        elif action == 'pick_up':
            if not has_ball and (r, c) == self.ball_pos:
                self.current_state = (r, c, True, has_key, door_open, kr, kc)
                return 10, self.current_state
            if not has_key and (r, c) == (kr, kc) and self.key_color == self.door_color:
                self.current_state = (r, c, has_ball, True, door_open, kr, kc)
                return 10, self.current_state
            return -1, self.current_state

        elif action == 'open_door':
            is_adjacent = (
                (r == self.door_row and c == self.wall_col - 1) or
                (r == self.door_row and c == self.wall_col + 1)
            )
            if is_adjacent and has_ball and has_key and not door_open:
                self.current_state = (r, c, has_ball, has_key, True, kr, kc)
                return 20, self.current_state
            return -1, self.current_state

        return -1, self.current_state

    def is_terminal(self):
        r, c = self.current_state[0], self.current_state[1]
        return (r, c) == self.goal_pos

    def reset(self):
        if self.randomize_start:
            start = self.start_positions[np.random.randint(len(self.start_positions))]
            self.agent_start = start

        # Randomize key position if multiple available
        if len(self.key_positions) > 1:
            self.key_pos = self.key_positions[np.random.randint(len(self.key_positions))]

        self.current_state = (
            self.agent_start[0], self.agent_start[1],
            False, False, False,
            self.key_pos[0], self.key_pos[1]
        )

    def get_states(self):
        states = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                if c == self.wall_col and r != self.door_row:
                    continue
                for has_ball in [False, True]:
                    for has_key in [False, True]:
                        for door_open in [False, True]:
                            for kr, kc in self.key_positions:
                                states.append((r, c, has_ball, has_key, door_open, kr, kc))
        return states

    def get_reward(self, action, state, new_state):
        r, c = new_state[0], new_state[1]
        if (r, c) == self.goal_pos:
            return 100
        return -1
