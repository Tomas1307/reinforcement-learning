from mdp import MDP
from loguru import logger
from gridworld_environment import GridWorld10x10
class ValueIteration:

    def __init__(self, mdp: MDP, discount: float = 0.9, iterations: int = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}
        for state in self.mdp.get_states():
            self.values[state] = 0.0

    def run_value_iteration(self):
        for i in range(self.iterations):
            new_values = {}
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    new_values[state] = 0.0
                else:
                    actions = self.mdp.get_possible_actions(state)
                    new_values[state] = max(
                        self.compute_qvalue_from_values(state, a) for a in actions
                    )
            self.values = new_values

    def get_value(self, state):
        return self.values.get(state, 0.0)

    def compute_qvalue_from_values(self, state, action) -> float:
        q_value = 0.0
        for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def compute_action_from_values(self, state):
        actions = self.mdp.get_possible_actions(state)
        if not actions:
            return None

        best_action = None
        best_qvalue = float('-inf')
        for action in actions:
            q = self.compute_qvalue_from_values(state, action)
            if q > best_qvalue:
                best_qvalue = q
                best_action = action
        return best_action

    def get_action(self, state):
        return self.compute_action_from_values(state)

    def get_qvalue(self, state, action):
        return self.compute_qvalue_from_values(state, action)

    def get_policy(self, state):
        if self.mdp.is_terminal(state):
            return None
        return self.compute_action_from_values(state)


if __name__ == "__main__":
    from bridge_environment import BridgeEnvironment

    # "GRIDWORLD" or "BRIDGE"
    ENVIRONMENT = "BRIDGE"  

    logger.add("value_iteration.log", rotation="500 MB", level="DEBUG")

    if ENVIRONMENT == "GRIDWORLD":
        env = GridWorld10x10()
        iterations_list = [5, 10, 15, 20, 30, 50]
        discount_values = [0.9]
    elif ENVIRONMENT == "BRIDGE":
        env = BridgeEnvironment()
        iterations_list = [10]
        discount_values = [0.9, 0.1]
    else:
        raise ValueError(f"Unknown environment: {ENVIRONMENT}")

    mdp = MDP(env)
    arrows = {'up': '  UP  ', 'down': ' DOWN ', 'left': ' LEFT ', 'right': 'RIGHT ', None: ' EXIT '}

    for discount in discount_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"ENVIRONMENT: {ENVIRONMENT}, DISCOUNT: {discount}")
        logger.info(f"{'='*60}\n")

        for n_iter in iterations_list:
            logger.info(f"=== Value Iteration con {n_iter} iteraciones ===")

            vi = ValueIteration(mdp, discount=discount, iterations=n_iter)
            vi.run_value_iteration()

            logger.info(f"--- Valores (iteraciones={n_iter}, discount={discount}) ---")
            for r in range(env.nrows):
                row = ''
                for c in range(env.ncols):
                    if env.board[r][c] == '#':
                        row += '  ####  '
                    else:
                        row += f'{vi.get_value((r, c)):+7.3f} '
                logger.info(row)

            logger.info(f"--- Politica (iteraciones={n_iter}, discount={discount}) ---")
            for r in range(env.nrows):
                row = ''
                for c in range(env.ncols):
                    if env.board[r][c] == '#':
                        row += ' #### '
                    else:
                        action = vi.get_policy((r, c))
                        row += f'{arrows[action]}'
                logger.info(row)

            logger.success(f"Value Iteration con {n_iter} iteraciones completado\n")