from dependencies import *
import pickle

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
    raise ValueError("didn't find valid goal")
    return p1_pos_or

def plan(layout_name, teams_list, n_actions=8):
    irl_weights = get_irl_weights(layout_name, teams_list)
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
    mlp = MediumLevelPlanner(overcooked_mdp, base_params_start_or)
    planner = MotionPlanner(overcooked_mdp)
    start_state = overcooked_mdp.get_standard_start_state()
    
    best_action = get_best_hl_action(overcooked_mdp, mlp, start_state, irl_weights, n_actions)
    goal_pos_or = get_goal_state(overcooked_mdp, mlp, start_state, best_action, planner)
    p1 = start_state.players[0]
    p1_pos_or = (p1.position, p1.orientation)
    # create motion plan to goal state
    action_plan, pos_and_or_path, cost = planner.get_plan(p1_pos_or, goal_pos_or)
    # TODO: do this in a loop, updating the players' & mdp state as appropriate

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
    ipdb.set_trace()

if __name__ == "__main__":
    # layout_name = "random0"
    # teams_list = [79]
    # plan(layout_name, teams_list)
    test_hl_action_planning("random1")

