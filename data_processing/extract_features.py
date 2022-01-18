import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, OvercookedGridworld, ObjectState
import ipdb

def json_eval(s):
    json_acceptable_string = s.replace("'", "\"")
    d = json.loads(json_acceptable_string)
    return d

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

N_ACTIONS = 9
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
    # 8: put onion in pot

    if (curr_item is None) and (next_item == "onion"):
        return 0
    elif curr_item is None and next_item == "dish":
        return 1
    elif curr_item is None and next_item == "soup":
        return 2
    elif curr_item == "dish" and next_item == "soup":
        return 3
    elif curr_item == "onion" and next_item is None:
        # split based on if onion was put in pot or counter
        if is_adjacent_to_serve(state, player, grid):
            return 8
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
    layout_name_to_data_name = {
        "random0": "random0",
        "random1": "coordination_ring",
        "simple": "cramped_room",
        "unident_s": "asymmetric_advantages",
        "random3": "random3"
    }
    data_name_to_layout_name = {
        "random0": "random0",
        "coordination_ring": "random1",
        "cramped_room": "simple",
        "asymmetric_advantages": "unident_s",
        "random3": "random3"
    }

    data = data.sort_values('time_elapsed')
    times = data.cur_gameloop
    grid = None
    times = [[], []]
    hl_actions = [[], []]
    state_encodings = []
    t = 0
    for index, row in data.iterrows():
        if grid is None:
            name = data_name_to_layout_name[row["layout_name"]]
            grid = OvercookedGridworld.from_grid(eval(row["layout"]), base_layout_params={"layout_name": name})
        # don't care about movement actions, we only need to split on "interact" actions
        state = json_eval(row["state"])
        # next_state = json_eval(row["next_state"])
        try:
            next_state = json_eval(data[data.cur_gameloop == (row.cur_gameloop+1)].state.to_numpy()[0])
        except:
            # TODO: make this less hacky, this will skip the last timestep since we have no next state
            continue
        
        # NOTE: loading PlayerState from dict errors if held_object key does not exist
        fix_held_obj(state)
        fix_held_obj(next_state)
        player_states = [PlayerState.from_dict(state["players"][player]) for player in range(2)] 
        # objects = create_obj_states(state["objects"])
        # NOTE: using from_dict fails because state dict contains extra key 'pot_explosion'
        # state_obj = OvercookedState(players=player_states, 
        #                             objects=objects, 
        #                             order_list=state["order_list"])
        state_obj = OvercookedState.from_dict(state)
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
            if a != -1:
                onehot[idx, a] = 1
        all_onehot.append(onehot)
    return all_onehot

def align_actions1(times, hl_actions):
    """
    Uses the stategy of backpropogating actions and making one data point per timestep
    """
    last_actions = [-1,-1]
    for player_idx in [0, 1]:
        if len(hl_actions[player_idx]) == 0:
            continue
        last_actions[player_idx] = times[player_idx][-1]
    # if neither player took any actions, return empty array
    if last_actions == [-1, -1]:
        return np.zeros((2, 0))

    # last_action = max(times[0][-1], times[1][-1])
    last_action = max(last_actions[0], last_actions[1])
    actions_full = -np.ones((2, last_action+1)) # +1 because of 0-indexing
    
    # fill with known actions
    for i, times_agent in enumerate(times):
        for j, t in enumerate(times_agent):
            actions_full[i, t] = hl_actions[i][j]

    # propogate known actions backward in time
    for player_idx in [0, 1]:
        if len(hl_actions[player_idx]) == 0:
            continue
        next_action = hl_actions[player_idx][-1]
        for i in range(last_actions[player_idx], -1, -1):
            if actions_full[player_idx][i] == -1:
                actions_full[player_idx][i] = next_action
            else:
                next_action = actions_full[player_idx][i]

    return actions_full


def process_pkl():
    filepath = "../human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle"
    pkl_file = open(filepath, "rb")
    df = pickle.load(pkl_file)

    trial_data_with_hl_actions = []

    for trial_id in df.trial_id.unique():
        for layout_id in df.layout_name.unique():
            trial_data = df[(df.trial_id == trial_id) & (df.layout_name == layout_id)]
            trial_data = trial_data.sort_values('cur_gameloop')
            if len(trial_data) == 0:
                continue

            times, hl_actions, state_encodings = extract_hl_actions(trial_data)
            all_hl_actions = align_actions1(times, hl_actions)
            onehot_hl_actions = onehot_encode_actions(all_hl_actions.astype(int))
            # merge for joint action
            onehot_hl_actions = np.hstack(onehot_hl_actions)
            # fill in the remaining timesteps with zeros (no high-level action taken)
            len_hl, n_actions = onehot_hl_actions.shape
            extras = np.zeros((len(trial_data)-len_hl, n_actions))
            onehot_hl_actions = np.vstack((onehot_hl_actions, extras))
            # save HL actions back into dataframes
            trial_data["high_level_action"] = list(onehot_hl_actions)
            trial_data_with_hl_actions.append(trial_data)
    
    # save data with high level actions added in
    new_df = pd.concat(trial_data_with_hl_actions)
    with open('../human_aware_rl/static/human_data/cleaned/2019_hh_trials_all_hl.pickle', 'wb') as handle:
        pickle.dump(new_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    process_pkl()
