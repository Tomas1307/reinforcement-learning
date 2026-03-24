import numpy as np
from loguru import logger


class GridWorld10x10:
    """10x10 GridWorld environment for TD Learning."""

    def __init__(self):
        self.nrows = 10
        self.ncols = 10
        self.initial_state = (0, 0)
        self.current_state = self.initial_state

        self._initialize_board()
        self._initialize_obstacles()

        self.actions = ['up', 'down', 'left', 'right']
        self.action_map = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        # Noise factors per action (unknown to the agent)
        # down/left: 0.2 noise (0.8 success)
        # up/right: 0.3 noise (0.7 success)
        self.noise = {
            'up': 0.3,
            'right': 0.3,
            'down': 0.2,
            'left': 0.2
        }

    def _initialize_board(self):
        self.board = [[' ' for _ in range(self.ncols)] for _ in range(self.nrows)]
        self.board[0][0] = 'S'
        self.board[4][5] = -1
        self.board[5][5] = 1
        self.board[7][4] = -1
        self.board[7][5] = -1

    def _initialize_obstacles(self):
        obstacles = [
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
            (3, 4), (4, 4), (5, 4), (6, 4), (8, 4)
        ]
        for r, c in obstacles:
            self.board[r][c] = '#'

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
        """Execute action with stochastic noise. Returns (reward, new_state)."""
        r, c = self.current_state

        if self.is_terminal():
            if action == 'exit':
                reward = self.board[r][c]
                return reward, self.current_state
            return 0, self.current_state

        # Stochastic execution based on noise factor
        noise = self.noise[action]
        other_actions = [a for a in self.actions if a != action]

        rand = np.random.random()
        if rand < (1 - noise):
            executed = action
        else:
            # Noise: equal probability among other 3 actions
            executed = np.random.choice(other_actions)

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

    def get_states(self):
        states = []
        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.board[r][c] != '#':
                    states.append((r, c))
        return states


class TDLearning:
    """
    Temporal Difference Learning TD(0) for estimating V^pi.
    Learns state values by following a given policy
    with unknown stochastic transitions.
    """

    def __init__(self, env: GridWorld10x10, policy: dict, alpha: float = 0.7, gamma: float = 0.96):
        self.env = env
        self.policy = policy  # dict: state (r, c) -> action
        self.alpha = alpha
        self.gamma = gamma

        # Initialize V(s) = 0 for all states
        self.V = {}
        for state in env.get_states():
            self.V[state] = 0.0

    def run_episode(self, max_steps: int = 1000):
        """
        Run one episode following the policy, updating V(s) at each step.
        Returns the number of steps taken.
        """
        self.env.reset()
        steps = 0

        while not self.env.is_terminal() and steps < max_steps:
            state = self.env.get_current_state()
            action = self.policy.get(state, 'right')  # fallback action

            reward, new_state = self.env.do_action(action)

            # TD(0) update: V(s) <- (1-alpha)*V(s) + alpha * [R + gamma * V(s')]
            old_v = self.V[state]
            self.V[state] = (1 - self.alpha) * self.V[state] + \
                            self.alpha * (reward + self.gamma * self.V[new_state])

            steps += 1

        # If we landed on a terminal state, update its value with the reward (no future)
        if self.env.is_terminal():
            state = self.env.get_current_state()
            r, c = state
            terminal_reward = self.env.board[r][c]
            self.V[state] = terminal_reward

        return steps

    def train(self, num_episodes: int = 1000):
        """
        Train for a given number of episodes.
        Returns history of V values for convergence analysis.
        """
        history = []

        for episode in range(num_episodes):
            steps = self.run_episode()

            # Save a snapshot of V for convergence analysis
            snapshot = dict(self.V)
            history.append(snapshot)

            if (episode + 1) % 100 == 0:
                logger.info(f"Episode {episode + 1}/{num_episodes} - Steps: {steps}")

        return history

    def derive_policy(self):
        """
        Derive a greedy policy from learned V(s).
        For each state, pick the action that leads to the highest V(s').
        Returns dict: state -> best action.
        """
        new_policy = {}
        actions = ['up', 'down', 'left', 'right']

        for state in self.env.get_states():
            r, c = state
            if isinstance(self.env.board[r][c], (int, float)):
                new_policy[state] = 'exit'
                continue

            best_action = None
            best_value = float('-inf')

            for action in actions:
                next_state = self.env._calculate_new_state(r, c, action)
                value = self.V.get(next_state, 0.0)
                if value > best_value:
                    best_value = value
                    best_action = action

            new_policy[state] = best_action

        return new_policy

    def print_values(self):
        """Print the value function as a grid."""
        print("\nValores:")
        for r in range(self.env.nrows):
            row_str = ""
            for c in range(self.env.ncols):
                if self.env.board[r][c] == '#':
                    row_str += "  #####  "
                else:
                    row_str += f" {self.V.get((r, c), 0.0):+.3f} "
            print(row_str)
        print()

    def print_policy(self, policy: dict):
        """Print a policy as a grid."""
        action_display = {
            'up': '  UP   ',
            'down': ' DOWN  ',
            'left': ' LEFT  ',
            'right': ' RIGHT ',
            'exit': ' EXIT  '
        }
        print("\nPolitica:")
        for r in range(self.env.nrows):
            row_str = ""
            for c in range(self.env.ncols):
                if self.env.board[r][c] == '#':
                    row_str += "  ####  "
                else:
                    a = policy.get((r, c), '?')
                    row_str += action_display.get(a, f" {a:^6s} ")
            print(row_str)
        print()


if __name__ == '__main__':
    env = GridWorld10x10()

    # Initial policy: move right, then down towards the +1 terminal state at (5,5)
    initial_policy = {}
    for r in range(env.nrows):
        for c in range(env.ncols):
            if isinstance(env.board[r][c], (int, float)):
                initial_policy[(r, c)] = 'exit'
            elif r < 5:
                initial_policy[(r, c)] = 'down'
            elif r > 5:
                initial_policy[(r, c)] = 'up'
            elif c < 5:
                initial_policy[(r, c)] = 'right'
            else:
                initial_policy[(r, c)] = 'left'

    logger.info("Training TD Learning with initial policy...")
    td = TDLearning(env, initial_policy, alpha=0.7, gamma=0.96)
    history = td.train(num_episodes=1000)

    td.print_values()

    # Derive and print improved policy
    new_policy = td.derive_policy()
    td.print_policy(new_policy)

    # Retrain with the new policy to refine
    logger.info("Retraining with derived policy...")
    td2 = TDLearning(env, new_policy, alpha=0.7, gamma=0.96)
    history2 = td2.train(num_episodes=1000)

    td2.print_values()
    final_policy = td2.derive_policy()
    td2.print_policy(final_policy)
