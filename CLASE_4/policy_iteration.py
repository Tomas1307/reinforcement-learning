from mdp import MDP


class PolicyIteration:

    def __init__(self, mdp: MDP, discount: float = 0.9, iterations: int = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = {}
        self.policy = {}

        # Initialize values to 0 and policy to a random action
        for state in self.mdp.get_states():
            self.values[state] = 0.0
            actions = self.mdp.get_possible_actions(state)
            if actions and not self.mdp.is_terminal(state):
                self.policy[state] = actions[0]  # Pick first action
            else:
                self.policy[state] = None

    def run_policy_iteration(self):
        """Runs the policy iteration algorithm for the given number of iterations."""
        for i in range(self.iterations):
            # Step 1: Policy Evaluation
            self.policy_evaluation()

            # Step 2: Policy Improvement
            policy_stable = self.policy_improvement()

            # If policy didn't change, we've converged
            if policy_stable:
                break

    def policy_evaluation(self, eval_iterations: int = 10):
        """
        Evaluate the current policy by computing values.
        Runs for a fixed number of iterations.
        """
        for _ in range(eval_iterations):
            new_values = {}
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    new_values[state] = 0.0
                else:
                    action = self.policy[state]
                    if action is None:
                        new_values[state] = 0.0
                    else:
                        new_values[state] = self.compute_qvalue_from_values(state, action)
            self.values = new_values

    def policy_improvement(self) -> bool:
        """
        Improve the policy by making it greedy w.r.t. current values.
        Returns True if policy is stable (didn't change), False otherwise.
        """
        policy_stable = True
        for state in self.mdp.get_states():
            if self.mdp.is_terminal(state):
                continue

            old_action = self.policy[state]
            new_action = self.compute_action_from_values(state)

            if old_action != new_action:
                policy_stable = False
                self.policy[state] = new_action

        return policy_stable

    def get_value(self, state):
        """Returns the value of a state."""
        return self.values.get(state, 0.0)

    def compute_qvalue_from_values(self, state, action) -> float:
        """Computes the Q-value of a state-action pair from current values."""
        q_value = 0.0
        for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def compute_action_from_values(self, state):
        """Returns the best action for a state based on computed values."""
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
        """Returns the policy action for a state (no exploration)."""
        return self.policy.get(state, None)

    def get_qvalue(self, state, action):
        """Returns the Q-value for a state-action pair."""
        return self.compute_qvalue_from_values(state, action)

    def get_policy(self, state):
        """Returns the action for a state, or None if no actions available."""
        if self.mdp.is_terminal(state):
            return None
        return self.policy.get(state, None)


if __name__ == "__main__":
    from loguru import logger
    from gridworld_environment import GridWorld10x10
    from bridge_environment import BridgeEnvironment

    # Choose environment: "GRIDWORLD" or "BRIDGE"
    ENVIRONMENT = "BRIDGE"

    logger.add("policy_iteration.log", rotation="500 MB", level="DEBUG")

    if ENVIRONMENT == "GRIDWORLD":
        env = GridWorld10x10()
        iterations_list = [5, 10, 15, 20]
        discount_values = [0.9]
    elif ENVIRONMENT == "BRIDGE":
        env = BridgeEnvironment()
        iterations_list = [5, 10]
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
            logger.info(f"=== Policy Iteration con {n_iter} iteraciones ===")

            pi = PolicyIteration(mdp, discount=discount, iterations=n_iter)
            pi.run_policy_iteration()

            # Log values grid
            logger.info(f"--- Valores (iteraciones={n_iter}, discount={discount}) ---")
            for r in range(env.nrows):
                row = ''
                for c in range(env.ncols):
                    if env.board[r][c] == '#':
                        row += '  ####  '
                    else:
                        row += f'{pi.get_value((r, c)):+7.3f} '
                logger.info(row)

            # Log policy grid
            logger.info(f"--- Politica (iteraciones={n_iter}, discount={discount}) ---")
            for r in range(env.nrows):
                row = ''
                for c in range(env.ncols):
                    if env.board[r][c] == '#':
                        row += ' #### '
                    else:
                        action = pi.get_policy((r, c))
                        row += f'{arrows[action]}'
                logger.info(row)

            logger.success(f"Policy Iteration con {n_iter} iteraciones completado\n")
