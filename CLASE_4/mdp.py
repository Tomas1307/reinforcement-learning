
from gridworld_environment import Environment

class MDP:
    def __init__(self, env: Environment):
        self.env = env

    def get_states(self):
        """Returns a list of all valid (non-obstacle) states."""
        states = []
        for r in range(self.env.nrows):
            for c in range(self.env.ncols):
                if self.env.board[r][c] != '#':
                    states.append((r, c))
        return states

    def get_possible_actions(self, state):
        """Returns the list of possible actions for a given state."""
        return self.env.get_possible_actions(state)

    def is_terminal(self, state):
        """Returns True if the state is a terminal state."""
        r, c = state
        return isinstance(self.env.board[r][c], (int, float))

    def get_reward(self, state, action, next_state):
        """Returns the reward for transitioning from state to next_state via action."""
        r, c = next_state
        if isinstance(self.env.board[r][c], (int, float)):
            return self.env.board[r][c]
        return 0.0

    def get_transition_states_and_probs(self, state, action):
        """
        Returns a list of (next_state, probability) tuples for taking
        action in state.
        """
        r, c = state

        if action == 'exit':
            return [(state, 1.0)]

        clockwise = self.env._get_clockwise_action(action)
        counterclockwise = self.env._get_counterclockwise_action(action)

        outcomes = [
            (action, self.env.action_success_prob),
            (clockwise, self.env.clockwise_prob),
            (counterclockwise, self.env.counterclockwise_prob),
            (None, self.env.stay_prob),
        ]

        # Aggregate probabilities by resulting state
        transition = {}
        for act, prob in outcomes:
            if act is None:
                next_state = (r, c)
            else:
                next_state = self.env._calculate_new_state(r, c, act)

            if next_state in transition:
                transition[next_state] += prob
            else:
                transition[next_state] = prob

        return list(transition.items())
