import numpy as np
import json
from loguru import logger


class QLearning:
    """
    Q-Learning: Off-policy TD control algorithm.

    Updates Q-values using: Q(s,a) <- (1-alpha)*Q(s,a) + alpha*[R + gamma*max_a' Q(s',a')]
    The key difference from SARSA: uses max over next actions (off-policy)
    instead of the action actually taken.
    """

    def __init__(self, env, alpha: float = 0.81, gamma: float = 0.96,
                 epsilon: float = 0.9, num_episodes: int = 1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        # Q-table: memory of the agent
        self.Q = {}
        for state in env.get_states():
            for action in env.get_possible_actions(state):
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        actions = self.env.get_possible_actions(state)

        if np.random.random() < self.epsilon:
            # Exploit: best action
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

    def step(self, action):
        """
        Execute an action in the environment.
        Returns (reward, done, new_state).
        """
        state = self.env.get_current_state()
        reward, new_state = self.env.do_action(action)
        done = self.env.is_terminal()
        return reward, done, new_state

    def get_reward(self, action, state, new_state):
        """Calculate the reward for a transition."""
        return self.env.get_reward(action, state, new_state)

    def run(self):
        """
        Execute the Q-Learning training loop.
        Returns rewards history.
        """
        rewards_history = []

        for episode in range(self.num_episodes):
            self.env.reset()
            state = self.env.get_current_state()
            total_reward = 0
            steps = 0

            while not self.env.is_terminal() and steps < 1000:
                # Choose action
                action = self.choose_action(state)

                # Execute step
                reward, done, new_state = self.step(action)
                total_reward += reward

                # Q-Learning update: use max Q(s', a') (off-policy)
                actions_next = self.env.get_possible_actions(new_state)
                max_q_next = max(
                    [self.Q.get((new_state, a), 0.0) for a in actions_next],
                    default=0.0
                )

                old_q = self.Q.get((state, action), 0.0)
                self.Q[(state, action)] = (1 - self.alpha) * old_q + \
                                           self.alpha * (reward + self.gamma * max_q_next)

                state = new_state
                steps += 1

            rewards_history.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                logger.info(f"Episode {episode + 1}/{self.num_episodes} - "
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

    def save_q_table(self, filepath: str):
        """Save the Q-table to a JSON file."""
        serializable = {}
        for (state, action), value in self.Q.items():
            key = f"{state}|{action}"
            serializable[key] = value
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath: str):
        """Load the Q-table from a JSON file."""
        with open(filepath, 'r') as f:
            serializable = json.load(f)
        self.Q = {}
        for key, value in serializable.items():
            parts = key.split('|')
            state_str = parts[0]
            action = parts[1]
            # Parse state tuple
            state = eval(state_str)
            self.Q[(state, action)] = value
        logger.info(f"Q-table loaded from {filepath}")

    def print_path(self):
        """Run a greedy episode and print the path."""
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
        if hasattr(self.env, 'print_grid'):
            self.env.print_grid(path=set(path))
        return path


if __name__ == '__main__':
    from cliff_walk_environment import CliffWalk

    env = CliffWalk()
    agent = QLearning(env, num_episodes=2000)

    logger.info("Training Q-Learning on Cliff-Walk...")
    rewards = agent.run()

    logger.info("Greedy path:")
    path = agent.print_path()
    print(f"Path length: {len(path) - 1}")
