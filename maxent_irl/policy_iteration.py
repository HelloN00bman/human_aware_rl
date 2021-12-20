from dependencies import *
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction
import pdb
# from state_featurization_for_irl import *




def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2, max_iters=1000):
    """
    Find the value function associated with a policy.
    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    for iteration in range(max_iters):
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = np.argmax(policy[s, :])
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

        if diff <= threshold:
            break

    return v


def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True, max_iters=10):
    """
    Find the optimal policy.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    #### Initialize random values and random policy
    v = np.random.normal(loc=0.0, scale=1.0, size=n_states)
    policy = np.random.choice(range(n_actions), size=(n_states, n_actions))


    policy_iteration_converged = False
    n_iterations = 0

    while not policy_iteration_converged:
        n_iterations += 1
        # policy evaluation
        v = value(policy, n_states, transition_probabilities, reward, discount,
              threshold=1e-2, max_iters=max_iters)

        policy_improvement_is_stable = True
        for s in range(n_states):
            a_old = np.argmax(policy[s, :])
            tp = transition_probabilities[s,:,:]
            a_new = np.argmax(np.dot(tp, reward + discount*v))
            policy[s, :] = np.dot(tp, reward + discount*v)

            if a_old != a_new:
                policy_improvement_is_stable = False

        if policy_improvement_is_stable or n_iterations > max_iters:
            policy_iteration_converged = True
            break

    print("policy shape", policy.shape)
    return policy
















