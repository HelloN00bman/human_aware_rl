import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, OvercookedGridworld, ObjectState

def get_player_states(state, grid):
    # NOTE: unused right now
    players = []
    for p in state["players"]:
        if "held_object" in p:
            players.append(PlayerState(p["position"], p["orientation"], p["held_object"]))
        else:
            players.append(PlayerState(p["position"], p["orientation"]))
    return players
    
def is_adjacent_to_serve(state, player, grid):
    serve_locs = grid.get_serving_locations()
    player_loc = state["players"][player]["position"]
    player_dir = state["players"][player]["orientation"]
    
    return any((np.add(player_loc, player_dir) == serve_locs).all(axis=1))

N_ACTIONS = 8
def get_hl_action(state, next_state, player, grid):
    # only getting the difference in held items
    # possible held items: onion, soup, dish
    # items = ["onion", "soup", "dish"]
    
    curr_item = state["players"][player]["held_object"]
    if curr_item is not None:
        curr_item = curr_item["name"]
    next_item = next_state["players"][player]["held_object"]
    if next_item is not None:
        next_item = next_item["name"]
    
    # NOTE: I think the only way to do this mapping is manually
    # possible high level actions: (encode with integer here, do one-hot encoding later)
    # 0: pick up onion
    # 1: pick pick up dish
    # 2: pick up soup (from counter)
    # 3: get cooked soup (from pot)
    # 4: put down onion
    # 5: put down dish
    # 6: put down soup (on counter)
    # 7: serve soup

    if (curr_item is None) and (next_item == "onion"):
        return 0
    elif curr_item is None and next_item == "dish":
        return 1
    elif curr_item is None and next_item == "soup":
        return 2
    elif curr_item == "dish" and next_item == "soup":
        return 3
    elif curr_item == "onion" and next_item is None:
        return 4
    elif curr_item == "dish" and next_item is None:
        return 5
    elif curr_item == "soup" and next_item is None:
        # split based on if soup was served or put on counter
        if is_adjacent_to_serve(state, player, grid):
            return 7
        return 6
    return None

def fix_held_obj(state):
    """
    Mutates state object to add key 'held_object' with value None if the key does not already exist
    """
    for player in state["players"]:
        if "held_object" not in player:
            player["held_object"] = None

def create_obj_states(objects):
    obj = {}
    for pos, o in objects.items():
        # NOTE: later code requires keys to be tuple not str, hence the eval(pos)
        obj[eval(pos)] = ObjectState.from_dict(o)
    return obj

def extract_hl_actions(data):
    data = data.sort_values('time_elapsed')
    times = data.cur_gameloop
    grid = None
    times = [[], []]
    hl_actions = [[], []]
    state_encodings = []
    t = 0
    for index, row in data.iterrows():
        if grid is None:
            grid = OvercookedGridworld.from_grid(eval(row["layout"]))
        # don't care about movement actions, we only need to split on "interact" actions
        state = eval(row["state"])
        next_state = eval(row["next_state"])
        
        # NOTE: loading PlayerState from dict errors if held_object key does not exist
        fix_held_obj(state)
        fix_held_obj(next_state)
        player_states = [PlayerState.from_dict(state["players"][player]) for player in range(2)] 
        objects = create_obj_states(state["objects"])
        # NOTE: using from_dict fails because state dict contains extra key 'pot_explosion'
        state_obj = OvercookedState(players=player_states, 
                                    objects=objects, 
                                    order_list=state["order_list"])
        state_encoding = grid.lossless_state_encoding(state_obj)
        state_encodings.append(state_encoding)
        for player in range(2):
            action = eval(row["joint_action"])[player]
            if action == "INTERACT":
                hl_action = get_hl_action(state, next_state, player, grid)
                if hl_action is not None:
                    times[player].append(t)
                    hl_actions[player].append(hl_action)
        t += 1
    return times, hl_actions, state_encodings

def onehot_encode_actions(hl_actions):
    all_onehot = []
    for actions in hl_actions:
        onehot = np.zeros((len(actions), N_ACTIONS))
        for idx, a in enumerate(actions):
            onehot[idx, a] = 1
        all_onehot.append(onehot)
    return all_onehot

def process_pkl(filepath="../human_aware_rl/data/human/anonymized/clean_train_trials.pkl"):
    pkl_file = open(filepath, "rb")
    df = pickle.load(pkl_file)

    for trial_id in df.workerid_num.unique():
        for layout_id in df.layout_name.unique():
            trial_data = df[(df.workerid_num == trial_id) & (df.layout_name == layout_id)]
            trial_data = trial_data.sort_values('cur_gameloop')

            times, hl_actions, state_encodings = extract_hl_actions(trial_data)
            onehot_hl_actions = onehot_encode_actions(hl_actions)

if __name__ == "__main__":
    process_pkl()
