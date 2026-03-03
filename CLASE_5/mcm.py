import numpy as np
from collections import defaultdict
from loguru import logger


class MCM:

    def __init__(self, env, discount: float = 0.9, epsilon: float = 0.3):
        self.env = env
        self.discount = discount
        self.epsilon = epsilon

        self.q_values = defaultdict(float)
        self.visit_counts = defaultdict(int)
        self.values = {}
        self.policy = {}
        self.convergence_history = []
        self.n_episodes = 0

        self._non_terminal_states = set()
        for r in range(env.nrows):
            for c in range(env.ncols):
                cell = env.board[r][c]
                if cell != '#' and not isinstance(cell, (int, float)):
                    self._non_terminal_states.add((r, c))

    def generate_episode(self, max_steps: int = 1000) -> list:
        episode = []
        self.env.reset()

        for _ in range(max_steps):
            state = self.env.get_current_state()

            if self.env.is_terminal():
                break

            action = self._select_action(state)
            reward, new_state = self.env.do_action(action)
            episode.append((state, action, reward))

        return episode

    def _select_action(self, state) -> str:
        actions = self.env.get_possible_actions(state)

        if not actions:
            return None

        if np.random.random() < self.epsilon:
            return np.random.choice(actions)

        best_action = None
        best_q = float('-inf')
        for a in actions:
            q = self.q_values.get((state, a), 0.0)
            if q > best_q:
                best_q = q
                best_action = a

        return best_action if best_action is not None else np.random.choice(actions)

    def update_from_episode(self, episode: list):
        G = 0.0
        visited = set()

        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G = reward + self.discount * G

            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                n = self.visit_counts[sa]
                self.q_values[sa] += (G - self.q_values[sa]) / (n + 1)
                self.visit_counts[sa] = n + 1

    def update_policy(self):
        states = set()
        for (state, action) in self.q_values:
            states.add(state)

        for state in states:
            actions = self.env.get_possible_actions(state)
            if not actions or actions == ['exit']:
                continue

            best_action = None
            best_q = float('-inf')
            for a in actions:
                q = self.q_values.get((state, a), 0.0)
                if q > best_q:
                    best_q = q
                    best_action = a

            self.policy[state] = best_action

    def update_values(self):
        states = set()
        for (state, action) in self.q_values:
            states.add(state)

        for state in states:
            actions = self.env.get_possible_actions(state)
            if actions == ['exit']:
                self.values[state] = 0.0
                continue

            max_q = float('-inf')
            for a in actions:
                q = self.q_values.get((state, a), 0.0)
                if q > max_q:
                    max_q = q
            self.values[state] = max_q if max_q != float('-inf') else 0.0

    def run(self, convergence_threshold: float = 0.005, check_interval: int = 100,
            patience: int = 3000, max_episodes: int = 500000) -> int:
        stable_count = 0
        checks_needed = patience // check_interval
        prev_values = {}
        min_coverage = 0.7

        for episode_num in range(1, max_episodes + 1):
            episode = self.generate_episode()
            self.update_from_episode(episode)
            self.n_episodes = episode_num

            if episode_num % check_interval == 0:
                self.update_policy()
                self.update_values()

                visited_states = set(self.policy.keys())
                coverage = len(visited_states) / len(self._non_terminal_states)

                max_change = 0.0
                for state, value in self.values.items():
                    old_value = prev_values.get(state, 0.0)
                    max_change = max(max_change, abs(value - old_value))

                self.convergence_history.append((episode_num, max_change, coverage))
                prev_values = dict(self.values)

                if coverage >= min_coverage and max_change < convergence_threshold:
                    stable_count += 1
                else:
                    stable_count = 0

                logger.debug(
                    f"Episodio {episode_num}: max_change={max_change:.6f}, "
                    f"cobertura={coverage:.1%}, estable={stable_count}/{checks_needed}"
                )

                if stable_count >= checks_needed:
                    logger.info(f"Convergencia despues de {episode_num} episodios "
                                f"(cobertura: {coverage:.1%})")
                    break

        self.update_policy()
        self.update_values()
        return self.n_episodes

    def get_value(self, state) -> float:
        return self.values.get(state, 0.0)

    def get_policy(self, state):
        r, c = state
        if isinstance(self.env.board[r][c], (int, float)):
            return None
        return self.policy.get(state, None)

    def get_qvalue(self, state, action) -> float:
        return self.q_values.get((state, action), 0.0)

    def print_values(self):
        logger.info("Valores:")
        for r in range(self.env.nrows):
            row = ''
            for c in range(self.env.ncols):
                if self.env.board[r][c] == '#':
                    row += '  ####  '
                else:
                    row += f'{self.get_value((r, c)):+7.3f} '
            logger.info(row)

    def print_policy(self):
        arrows = {
            'up': '  UP  ', 'down': ' DOWN ', 'left': ' LEFT ',
            'right': 'RIGHT ', None: ' EXIT '
        }
        logger.info("Politica:")
        for r in range(self.env.nrows):
            row = ''
            for c in range(self.env.ncols):
                if self.env.board[r][c] == '#':
                    row += ' #### '
                else:
                    action = self.get_policy((r, c))
                    row += f'{arrows.get(action, " NONE ")}'
            logger.info(row)


if __name__ == "__main__":
    from gridworld_environment import GridWorld10x10

    logger.add("mcm.log", rotation="500 MB", level="DEBUG")

    env = GridWorld10x10()
    mcm = MCM(env, discount=0.9, epsilon=0.3)

    n_episodes = mcm.run()
    logger.success(f"MCM finalizado en {n_episodes} episodios")

    mcm.print_values()
    mcm.print_policy()
