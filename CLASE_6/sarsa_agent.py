import numpy as np
from loguru import logger
from cliff_walk_environment import CliffWalk


class SARSA:
    """
    SARSA: On-policy TD control algorithm.

    Updates Q-values using: Q(s,a) <- (1-alpha)*Q(s,a) + alpha*[R + gamma*Q(s',a')]
    where a' is the action actually chosen in s' (on-policy).
    """

    def __init__(self, env, epsilon: float = 0.9, gamma: float = 0.96, alpha: float = 0.81):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # Initialize Q(s, a) = 0 for all state-action pairs
        self.Q = {}
        for state in env.get_states():
            for action in env.get_possible_actions(state):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy.
        With probability epsilon: exploit (best action).
        With probability 1-epsilon: explore (random action).
        """
        actions = self.env.get_possible_actions(state)

        if np.random.random() < self.epsilon:
            # Exploit: choose best action
            best_action = None
            best_value = float('-inf')
            for action in actions:
                q_val = self.Q.get((state, action), 0.0)
                if q_val > best_value:
                    best_value = q_val
                    best_action = action
            return best_action
        else:
            # Explore: random action
            return np.random.choice(actions)

    def action_function(self, state1, action1, reward, state2, action2):
        """
        SARSA update rule.
        Q(s1, a1) <- (1-alpha)*Q(s1, a1) + alpha*[R + gamma*Q(s2, a2)]
        """
        old_q = self.Q.get((state1, action1), 0.0)
        next_q = self.Q.get((state2, action2), 0.0)

        self.Q[(state1, action1)] = (1 - self.alpha) * old_q + \
                                     self.alpha * (reward + self.gamma * next_q)

    def run_episode(self, max_steps: int = 1000):
        """
        Run one SARSA episode.
        Returns (total_reward, steps).
        """
        self.env.reset()
        state = self.env.get_current_state()
        action = self.choose_action(state)

        total_reward = 0
        steps = 0

        while not self.env.is_terminal() and steps < max_steps:
            # (1) Execute action, get reward and next state
            reward, next_state = self.env.do_action(action)
            total_reward += reward

            # (2) Choose next action for the next state
            next_action = self.choose_action(next_state)

            # (3) Update Q-value with SARSA
            self.action_function(state, action, reward, next_state, next_action)

            # (4) Move to next state-action pair
            state = next_state
            action = next_action
            steps += 1

        return total_reward, steps

    def train(self, num_episodes: int = 500):
        """Train the agent for a number of episodes."""
        rewards_history = []

        for episode in range(num_episodes):
            total_reward, steps = self.run_episode()
            rewards_history.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                logger.info(f"Episode {episode + 1}/{num_episodes} - "
                           f"Avg Reward (last 100): {avg_reward:.1f} - Steps: {steps}")

        return rewards_history

    def get_policy(self):
        """Extract the greedy policy from the Q-table."""
        policy = {}
        for state in self.env.get_states():
            actions = self.env.get_possible_actions(state)
            best_action = None
            best_value = float('-inf')
            for action in actions:
                q_val = self.Q.get((state, action), 0.0)
                if q_val > best_value:
                    best_value = q_val
                    best_action = action
            policy[state] = best_action
        return policy

    def print_policy(self):
        """Print the learned policy as a grid."""
        policy = self.get_policy()
        action_display = {
            'up': '  UP  ',
            'down': ' DOWN ',
            'left': ' LEFT ',
            'right': 'RIGHT '
        }
        print("\nPolitica SARSA:")
        for r in range(self.env.nrows):
            row_str = ""
            for c in range(self.env.ncols):
                state = (r, c)
                if state == self.env.goal:
                    row_str += " GOAL "
                elif state in self.env.cliff:
                    row_str += " XXXX "
                else:
                    a = policy.get(state, '?')
                    row_str += action_display.get(a, f" {a:^5s} ")
            print(row_str)
        print()

    def print_path(self):
        """Run a greedy episode and print the path taken."""
        self.env.reset()
        path = [self.env.get_current_state()]

        old_epsilon = self.epsilon
        self.epsilon = 1.0  # fully greedy

        steps = 0
        while not self.env.is_terminal() and steps < 100:
            state = self.env.get_current_state()
            action = self.choose_action(state)
            _, next_state = self.env.do_action(action)
            path.append(next_state)
            steps += 1

        self.epsilon = old_epsilon
        self.env.print_grid(path=set(path))
        return path


if __name__ == '__main__':
    env = CliffWalk()

    logger.info("Training SARSA agent on Cliff-Walk...")
    agent = SARSA(env, epsilon=0.9, gamma=0.96, alpha=0.81)
    rewards = agent.train(num_episodes=500)

    agent.print_policy()

    logger.info("Greedy path after training:")
    agent.print_path()
