import numpy as np
from loguru import logger


class LockedDoorEnv:
    """
    Locked-Door environment.

    Two rooms separated by a locked door. The agent must:
    1. Pick up the ball blocking the door
    2. Pick up the key (same color as door)
    3. Open the door
    4. Reach the goal in the right room

    State: (row, col, has_ball, has_key, door_open)

    Grid layout (default):
        Left room (cols 0-4), Wall at col 5 with door at row 2, Right room (cols 6-10)

        . . . . . | . . . . .
        . . . . . | . . . . .
        . . . . . D . . . . .
        . . . . . | . . . . .
        . . . . . | . . . . .

    A = Agent start (random or fixed)
    K = Key (blue)
    B = Ball (blocking door)
    D = Door (blue, locked)
    G = Goal (in right room)
    """

    def __init__(self, nrows=5, ncols=11, wall_col=5, door_row=2,
                 agent_start=None, key_pos=None, ball_pos=None, goal_pos=None,
                 key_color='blue'):
        self.nrows = nrows
        self.ncols = ncols
        self.wall_col = wall_col
        self.door_row = door_row
        self.key_color = key_color
        self.door_color = 'blue'

        # Default positions (all in left room)
        self.agent_start = agent_start or (0, 0)
        self.key_pos = key_pos or (4, 3)
        self.ball_pos = ball_pos or (2, 4)  # next to door
        self.goal_pos = goal_pos or (4, 10)

        # State: (row, col, has_ball, has_key, door_open)
        self.current_state = (self.agent_start[0], self.agent_start[1], False, False, False)

        # Actions
        self.movement_actions = ['up', 'down', 'left', 'right']
        self.special_actions = ['pick_up', 'open_door']
        self.actions = self.movement_actions + self.special_actions

        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        return self.actions

    def _is_wall(self, r, c):
        """Check if position is a wall."""
        if r < 0 or r >= self.nrows or c < 0 or c >= self.ncols:
            return True
        # Wall column (except door)
        if c == self.wall_col:
            if r == self.door_row:
                # Door: only passable if open
                _, _, _, _, door_open = self.current_state
                return not door_open
            return True
        return False

    def do_action(self, action):
        """Execute action. Returns (reward, new_state)."""
        r, c, has_ball, has_key, door_open = self.current_state

        if action in self.movement_actions:
            dr, dc = self.action_map[action]
            new_r, new_c = r + dr, c + dc

            # Check bounds and walls
            if self._is_wall(new_r, new_c):
                self.current_state = (r, c, has_ball, has_key, door_open)
                return -1, self.current_state

            self.current_state = (new_r, new_c, has_ball, has_key, door_open)

            # Check if reached goal
            if (new_r, new_c) == self.goal_pos:
                return 100, self.current_state

            return -1, self.current_state

        elif action == 'pick_up':
            # Try to pick up ball
            if not has_ball and (r, c) == self.ball_pos:
                self.current_state = (r, c, True, has_key, door_open)
                return 10, self.current_state

            # Try to pick up key
            if not has_key and (r, c) == self.key_pos:
                # Key must match door color
                if self.key_color == self.door_color:
                    self.current_state = (r, c, has_ball, True, door_open)
                    return 10, self.current_state

            # Invalid pick up
            return -1, self.current_state

        elif action == 'open_door':
            # Must be adjacent to door, have ball removed, and have key
            is_adjacent = (
                (r == self.door_row and c == self.wall_col - 1) or
                (r == self.door_row and c == self.wall_col + 1)
            )
            if is_adjacent and has_ball and has_key and not door_open:
                self.current_state = (r, c, has_ball, has_key, True)
                return 20, self.current_state

            # Invalid open
            return -1, self.current_state

        return -1, self.current_state

    def is_terminal(self):
        r, c, _, _, _ = self.current_state
        return (r, c) == self.goal_pos

    def reset(self):
        self.current_state = (self.agent_start[0], self.agent_start[1], False, False, False)

    def get_states(self):
        """Generate all possible states."""
        states = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                # Skip wall positions (except door when open)
                if c == self.wall_col and r != self.door_row:
                    continue
                for has_ball in [False, True]:
                    for has_key in [False, True]:
                        for door_open in [False, True]:
                            states.append((r, c, has_ball, has_key, door_open))
        return states

    def get_reward(self, action, state, new_state):
        """Calculate reward for a transition."""
        r, c, _, _, _ = new_state
        if (r, c) == self.goal_pos:
            return 100
        return -1

    def print_grid(self, path=None):
        """Print the current state of the grid."""
        r, c, has_ball, has_key, door_open = self.current_state
        print(f"Agent: ({r},{c}) | Ball: {'picked' if has_ball else str(self.ball_pos)} | "
              f"Key: {'picked' if has_key else str(self.key_pos)} | "
              f"Door: {'OPEN' if door_open else 'LOCKED'}")

        for row in range(self.nrows):
            row_str = ""
            for col in range(self.ncols):
                if (row, col) == (r, c):
                    row_str += " A "
                elif (row, col) == self.goal_pos:
                    row_str += " G "
                elif not has_ball and (row, col) == self.ball_pos:
                    row_str += " B "
                elif not has_key and (row, col) == self.key_pos:
                    row_str += " K "
                elif col == self.wall_col:
                    if row == self.door_row:
                        row_str += " D " if not door_open else " _ "
                    else:
                        row_str += " | "
                elif path and (row, col) in path:
                    row_str += " . "
                else:
                    row_str += " - "
            print(row_str)
        print()


if __name__ == '__main__':
    env = LockedDoorEnv()
    env.print_grid()
    print(f"States: {len(env.get_states())}")
    print(f"Actions: {env.actions}")
