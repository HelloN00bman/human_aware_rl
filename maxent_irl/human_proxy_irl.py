from dependencies import *
import pickle

from overcooked_ai_py.planning.search import Graph
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner, MediumLevelPlanner
from irl_test_envs import get_start_state
import ipdb

def get_irl_weights(layout_name, teams_list):
    filename = './irl_weights_'+layout_name + '_' + str(teams_list) + '.pkl'
    with open(filename, 'rb') as handle:
        irl_weights = pickle.load(handle)
    return irl_weights

def get_best_hl_action(mdp, mlp, state, irl_weights, n_actions):
    state_features = mdp.featurize_state_for_irl(state, mlp, [])

    action_rewards = []
    all_actions = [(i, j) for i in range(n_actions) for j in range(n_actions)]
    for i, j in all_actions:
        onehot_action = np.concatenate([np.eye(n_actions)[i], np.eye(n_actions)[j]])
        features = np.hstack((state_features, onehot_action))
        action_rewards.append(features.T @ irl_weights)
    # TODO: figure out how to deal with the fact that this is reasoning over joint actions,
    # even though we only want to pick actions for one of the agents (maybe we need to change
    # the IRL features to include the future HL action of the human but the previous HL action of
    # the other agent)
    best_action = all_actions[np.argmax(action_rewards)][0]

    return best_action

def get_goal_state(mdp, mlp, state, hl_action, planner):
    p1 = state.players[0]
    p1_pos_or = (p1.position, p1.orientation)
    # TODO: check for validity of goal_locs outside of if statement
    if hl_action == 0:
        # pick up onion (from dispenser or counter)
        counter_objects = mdp.get_counter_objects_dict(state)
        goal_locs = mlp.ml_action_manager.pickup_onion_actions(state, counter_objects)
        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
    elif hl_action == 1:
        # pick up dish (from dispenser or counter)
        counter_objects = mdp.get_counter_objects_dict(state)
        goal_locs = mlp.ml_action_manager.pickup_dish_actions(state, counter_objects)

        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
    elif hl_action == 2:
        # pick up soup (from counter)
        counter_objects = mdp.get_counter_objects_dict(state)
        goal_locs = mlp.ml_action_manager.pickup_counter_soup_actions(state, counter_objects)
        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
    elif hl_action == 3:
        # get cooked soup (from pot)
        # TODO: make sure this action is only picked we're already holding a plate
        pot_states_dict = mdp.get_pot_states(state)
        goal_locs = mlp.ml_action_manager.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
        pass
    elif hl_action in [4, 5, 6]:
        # planning to counter location is independent of held object
        # 4: put onion down on counter
        # 5: put dish down on counter
        # 6: put soup down on counter
        # TODO: do we need additional intelligence to determine which type of counter to place on?
        # e.g. a shared counter location vs others
        # TODO: decide if we should collapse these into a single action: "put down held object"
        goal_locs = mlp.ml_action_manager.place_obj_on_counter_actions(state)

        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
    elif hl_action == 7:
        # serve soup (plan to serving location)
        goal_locs = mlp.ml_action_manager.deliver_soup_actions()
        
        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
    elif hl_action == 8:
        # put down onion (into pot)
        pot_states_dict = mdp.get_pot_states(state)
        goal_locs = mlp.ml_action_manager.put_onion_in_pot_actions(pot_states_dict)

        # check if any of these locations are valid
        for goal_pos_or in goal_locs:
            if planner.is_valid_motion_start_goal_pair(p1_pos_or, goal_pos_or):
                return goal_pos_or
    else:
        raise ValueError(f"unrecognized high level action {hl_action}")

    # TODO: if we didn't find a valid goal position, just return the player's current position back
    # raise ValueError("didn't find valid goal")
    return p1_pos_or

def graph_from_grid_include_partner(planner, p2_pos_or):
    """Creates a graph adjacency matrix from an Overcooked MDP class."""
    # State decoder is a dictionary.
    # For all valid player positions and orientations, insert into the state decoder.
    # State decoder takes form -- Counter ID: player ( position, orientation )
    # Position encoder takes form -- player ( position, orientation ): Counter ID
    # Max counter ID = num of graph nodes

    state_decoder = {}
    for state_index, motion_state in enumerate(planner.mdp.get_valid_player_positions_and_orientations()):
        state_decoder[state_index] = motion_state

    pos_encoder = {motion_state: state_index for state_index, motion_state in state_decoder.items()}
    num_graph_nodes = len(state_decoder)

    adjacency_matrix = np.zeros((num_graph_nodes, num_graph_nodes))
    for state_index, start_motion_state in state_decoder.items():
        # For each possible next state that the player can be in given an action.
        # action = id, successor_motion_state = (new_pos, new_orientation)
        if start_motion_state[0] != p2_pos_or[0]: # don't include other player's state in graph
            for action, successor_motion_state in planner._get_valid_successor_motion_states(start_motion_state):
                adj_pos_index = pos_encoder[successor_motion_state]
                if successor_motion_state[0] != p2_pos_or[0]: # don't include other player's state as valid transition
                    adjacency_matrix[state_index][adj_pos_index] = planner._graph_action_cost(action)
                    # An action can take you from one state to the next, given the cost of the action.

    return Graph(adjacency_matrix, pos_encoder, state_decoder)

def plan(layout_name, teams_list, n_actions=8):
    irl_weights = get_irl_weights(layout_name, teams_list)
    layout_name = "random1" # TODO: get rid of this after debugging
    # construct the gridworld
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }
    mlp = MediumLevelPlanner.from_action_manager_file("random1_am.pkl")
    mlp.ml_action_manager.counter_drop = overcooked_mdp.terrain_pos_dict['X'] # TODO: save ml_action_manager file to include this
    # mlp = MediumLevelPlanner(overcooked_mdp, base_params_start_or)
    planner = MotionPlanner(overcooked_mdp)
    state = overcooked_mdp.get_standard_start_state()

    action_plan = []
    for t in range(100): # we have no other termination condition here
        if action_plan == []:
            best_action = get_best_hl_action(overcooked_mdp, mlp, state, irl_weights, n_actions)
            print(best_action)
            goal_pos_or = get_goal_state(overcooked_mdp, mlp, state, best_action, planner)
            p1 = state.players[0]
            p1_pos_or = (p1.position, p1.orientation)
            p2 = state.players[1]
            p2_pos_or = (p2.position, p2.orientation)
            # create motion plan to goal state
            
            graph = graph_from_grid_include_partner(planner, p2_pos_or)
            node_path = graph.get_node_path(p1_pos_or, goal_pos_or)
            positions_plan = [state_node[0] for state_node in node_path[1:]]
            action_plan, pos_and_or_path, plan_length = planner.action_plan_from_positions(positions_plan, p1_pos_or, goal_pos_or)

        if action_plan == []:
            action = (0,0)
        else:
            action = action_plan.pop(0)
        # TODO: decide what we should do if the other agent is in the way of the human (or the human 
        # is otherwise blocked from executing this plan)
        p2_action = (0,0)
        state, sparse_reward, shaped_reward = overcooked_mdp.get_state_transition(state, (action, p2_action))
    ipdb.set_trace()

def test_hl_action_planning(layout_name="random0"):
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }
    # mlp = MediumLevelPlanner(overcooked_mdp, base_params_start_or)
    mlp = MediumLevelPlanner.from_action_manager_file("random1_am.pkl")
    mlp.ml_action_manager.counter_drop = overcooked_mdp.terrain_pos_dict['X'] # TODO: save ml_action_manager file to include this
    planner = MotionPlanner(overcooked_mdp, counter_goals=overcooked_mdp.terrain_pos_dict['X'])
    start_state = get_start_state("put_down_onion", layout_name=layout_name)
    
    goal_pos = get_goal_state(overcooked_mdp, mlp, start_state, 8, planner)

    print(overcooked_mdp.state_string(start_state))
    print(goal_pos)

if __name__ == "__main__":
    layout_name = "random0"
    teams_list = [79]
    plan(layout_name, teams_list)
