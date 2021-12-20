from dependencies import *
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction
import pdb
from state_featurization_for_irl import *
from fixed_strat_state_featurization import run_fixed_featurization, run_data_featurization, run_data_featurization_reduced
# from numba import cuda, float32
from multiprocessing.pool import ThreadPool as Pool

from itertools import product

import numpy as np
import numpy.random as rn

import value_iteration
import policy_iteration

# @cuda.jit
def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, max_policy_iters):
    """
    Find the reward function for the given trajectories.
    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    alpha = rn.uniform(size=(d_states,))

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories)

    # Gradient descent on alpha.
    for i in range(epochs):
        print("IRL Iteration = i: {}".format(i))
        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf_multiprocessed(n_states, r, n_actions, discount,
                                         transition_probability, trajectories, max_policy_iters)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        alpha += learning_rate * grad
        alpha /= sum(alpha)
        print(f"alpha at iteration={i} = {alpha}")


    return feature_matrix.dot(alpha).reshape((n_states,)), alpha

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.
    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.
    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, action in trajectory:
            feature_expectations += feature_matrix[state]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories, max_iters=100):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.
    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """
    pool_size = 64

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    print("FINDING POLICY")
    policy = policy_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount, max_iters)
    print("policy found")

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories

    print("Getting expected SVF")
    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    # for t in range(1, trajectory_length):
    #     expected_svf[:, t] = 0
    #     for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
    #         expected_svf[k, t] += (expected_svf[i, t-1] *
    #                               policy[i, j] * # Stochastic policy
    #                               transition_probability[i, j, k])
    for t in range(1, int(trajectory_length/4)):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
        # for i, j, k in product(range(int(n_states/10)), range(int(n_actions/10)), range(int(n_states/10))):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])
    print("Found expected SVF")
    return expected_svf.sum(axis=1)



def find_expected_svf_multiprocessed(n_states, r, n_actions, discount,
                      transition_probability, trajectories, max_iters=100):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.
    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """
    pool_size = 64

    # define worker function before a Pool is instantiated
    def worker(i, t, j, k):
        try:
            expected_svf[k, t] += (expected_svf[i, t - 1] *
                                   policy[i, j] *  # Stochastic policy
                                   transition_probability[i, j, k])
        except:
            print('error with item')

    n_trajectories = trajectories.shape[0]

    for traj in trajectories:
        print(len(traj))

    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    print("FINDING POLICY")
    policy = policy_iteration.find_policy(n_states, n_actions,
                                          transition_probability, r, discount, max_iters)
    print("policy found")

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count / n_trajectories

    print("Getting expected SVF multiprocessed")
    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    # for t in range(1, trajectory_length):
    #     expected_svf[:, t] = 0
    #     for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
    #         expected_svf[k, t] += (expected_svf[i, t-1] *
    #                               policy[i, j] * # Stochastic policy
    #                               transition_probability[i, j, k])
    # for t in range(1, int(trajectory_length / 4)):
    #     expected_svf[:, t] = 0
    #     for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
    #         # for i, j, k in product(range(int(n_states/10)), range(int(n_actions/10)), range(int(n_states/10))):
    #         expected_svf[k, t] += (expected_svf[i, t - 1] *
    #                                policy[i, j] *  # Stochastic policy
    #                                transition_probability[i, j, k])

    pool = Pool(pool_size)

    for t in range(1, int(trajectory_length / 1)):
        if t % 100 == 0:
            print("t = ", t)
        expected_svf[:, t] = 0
        for i, j, k in product(range(int(n_states/1)), range(int(n_actions/1)), range(int(n_states/1))):
            pool.apply_async(worker, (i, t, j, k,))

    pool.close()
    pool.join()

    print("Found expected SVF")
    return expected_svf.sum(axis=1)


def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.
    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).
    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.
    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy = value_iteration.find_policy(n_states, n_actions,
        transition_probability, reward, discount)
    value = value_iteration.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd



def main():
    a0_type, a1_type = 'SP', 'SP'

    state_seq, action_seq, feature_seq, transition_matrix, state_idx_to_state, state_tuple_to_state_idx, state_reward_list, feature_matrix, trajectories = run_data_featurization_reduced(a0_type, a1_type)
    print("state_seq shape", state_seq.shape)
    print('action_seq shape', action_seq.shape)
    print("feature_seq.shape = ", feature_seq.shape)
    print("state_reward_list", state_reward_list.shape)
    print("transition_matrix", transition_matrix.shape)
    print('feature_matrix', feature_matrix.shape)
    # print(feature_seq)

    n_states = transition_matrix.shape[0]
    n_actions = transition_matrix.shape[1]
    transition_probability = transition_matrix
    reward = state_reward_list
    discount = 0.9
    epochs = 5
    learning_rate = 0.1
    max_policy_iters = 1000

    irl_results, feature_weights = irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, max_policy_iters)

    print("found irl_results = ", irl_results)
    print("feature_weights", feature_weights)


def see_max_ent_results():
    dp_6 = np.array([0.09302326, 0.17054264, 0.07751938, 0.07751938, 0.50387597, 0.07751938])
    dp_5 = np.array([0.1875,  0.34375, 0.15625, 0.15625, 0.15625])

    sp_5 = np.array([0.18367347, 0.32653062, 0.1632653,  0.16326531, 0.1632653])
    sp_6 = np.array([1.83673467e-01, 3.26530614e-01, 1.63265300e-01, 1.63265313e-01, 1.81663590e-09, 1.63265304e-01])

    scale_factor = 15
    print("DP 6: ", dp_6 * scale_factor)
    print("SP 6: ", sp_6 * scale_factor)


if __name__ == '__main__':
    main()
    # see_max_ent_results()



