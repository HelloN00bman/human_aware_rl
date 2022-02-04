from dependencies import *
import pickle

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner, MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction

# Test Action 0: pick up onion
# Test Action 1: pick up dish
def default_start_state(layout_name="random0"):
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }

    start_state = overcooked_mdp.get_standard_start_state()


    return start_state


# Test Action 2: pick up soup (from counter)
def pick_up_soup_from_counter_start_state(layout_name="random0"):
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }

    start_state = overcooked_mdp.get_standard_start_state()

    counter_location = overcooked_mdp.get_counter_locations()[1]

    # output_state = OvercookedState.from_players_pos_and_or(players_pos_and_or, order_list)

    obj_dict = {'name': "soup", 'position': counter_location, '_ingredients': ['onion', 'onion', 'onion'],
                'cook_time': 20}

    soup_obj = ObjectState.from_dict(obj_dict)

    start_state.add_object(soup_obj)

    return start_state


# Test Action 3: get cooked soup (from pot)
def get_soup_from_pot_start_state(layout_name="random0"):
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }

    start_state = overcooked_mdp.get_standard_start_state()

    pot_location = overcooked_mdp.get_pot_locations()[0]

    # output_state = OvercookedState.from_players_pos_and_or(players_pos_and_or, order_list)

    obj_dict = {'name': "soup", 'position': pot_location, '_ingredients': ['onion', 'onion', 'onion'], 'cook_time': 20}

    soup_obj = ObjectState.from_dict(obj_dict)

    start_state.add_object(soup_obj)

    return start_state


# Test Action 4-6: put down object (4. onion, 5. dish, 6. soup) on counter
def put_down_object_start_state(layout_name="random0", holding="onion"):
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }

    start_state = overcooked_mdp.get_standard_start_state()

    pot_location = overcooked_mdp.get_pot_locations()[0]

    players = start_state.players
    for player in players:
        # player.set_object(new_state.remove_object(i_pos))
        player_position = player.position

        if holding == "onion":
            holding_obj = {'name': "onion", 'position': player_position}
            player.set_object(ObjectState.from_dict(holding_obj))

        elif holding == "dish":
            holding_obj = {'name': "onion", 'position': player_position}
            player.set_object(ObjectState.from_dict(holding_obj))

        elif holding == "soup":
            holding_obj = {'name': "soup", 'position': player_position, '_ingredients': ['onion', 'onion', 'onion'], 'cook_time': 20}
            player.set_object(ObjectState.from_dict(holding_obj))


    return start_state


# Test Action 7: Serve Soup
def serve_soup_start_state(layout_name="random0"):
    overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }

    start_state = overcooked_mdp.get_standard_start_state()

    players = start_state.players
    for player in players:
        player_position = player.position

        holding_obj = {'name': "soup", 'position': player_position, '_ingredients': ['onion', 'onion', 'onion'],
                           'cook_time': 20}
        player.set_object(ObjectState.from_dict(holding_obj))

    return start_state

def get_start_state(action, layout_name="random0"):
    if action == "pick_up_onion":
        state = default_start_state(layout_name)
    elif action == "pick_up_dish":
        state = default_start_state(layout_name)
    elif action == "pick_up_soup":
        state = pick_up_soup_from_counter_start_state(layout_name)
    elif action == "get_cooked_soup":
        state = get_soup_from_pot_start_state(layout_name)
    elif action == "put_down_onion":
        state = put_down_object_start_state(layout_name, holding="onion")
    elif action == "put_down_dish":
        state = put_down_object_start_state(layout_name, holding="dish")
    elif action == "put_down_soup":
        state = put_down_object_start_state(layout_name, holding="soup")
    elif action == "serve_soup":
        state = serve_soup_start_state(layout_name)
    else:
        state = default_start_state(layout_name)
    return state

if __name__ == "__main__":
    test_actions = ["pick_up_onion", "pick_up_dish", "pick_up_soup", "get_cooked_soup", "put_down_onion", "put_down_dish",
                    "put_down_soup", "serve_soup"]

    for action in test_actions:
        test_state = get_start_state(action, layout_name="random0")
        print(f"ACTION: {action}, test_state {test_state}")