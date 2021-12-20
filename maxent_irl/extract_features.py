# from dependencies import *
from dependencies import *

from process_data import json_eval, import_2019_data

def f_p1(t, a, p1_data):
    t_partial = t[int(a) - 1:int(a)]
    #     print('t_input', t)
    return [p1_data[int(j)]['position'][0] for j in t_partial], [p1_data[int(j)]['position'][1] for j in t_partial]


def f_p2(t, a, p2_data):
    t_partial = t[int(a) - 1:int(a)]
    return [p2_data[int(j)]['position'][0] for j in t_partial], [p2_data[int(j)]['position'][1] for j in t_partial]


def arrow_p1(t, a, p1_data):
    #     t_partial = t[int(a)-1:int(a)+1]
    #     print('t_input', t)
    return p1_data[int(a)]['position'][0], p1_data[int(a)]['position'][1], \
           p1_data[int(a)]['orientation'][0], p1_data[int(a)]['orientation'][1]


def arrow_p2(t, a, p2_data):
    return p2_data[int(a)]['position'][0], p2_data[int(a)]['position'][1], \
           p2_data[int(a)]['orientation'][0], p2_data[int(a)]['orientation'][1]


def held_p1(t, a, p1_data):
    return p1_data[int(a)]['position'][0], p1_data[int(a)]['position'][1], \
           p1_data[int(a)]['orientation'][0], p1_data[int(a)]['orientation'][1]


def held_p2(t, a, p2_data):
    return p2_data[int(a)]['position'][0], p2_data[int(a)]['position'][1], \
           p2_data[int(a)]['orientation'][0], p2_data[int(a)]['orientation'][1]


def world_obj(t, a, objects_data):
    obj_world = objects_data[int(a)]
    if len(obj_world) == 0:
        return []
    objects_list = []
    for i in range(len(obj_world)):
        obj = obj_world[i]
        name = obj_world[i]['name']
        position = obj_world[i]['position']
        if name == 'onion':
            color = 'y'
            objects_list.append((position[0], position[1], name, color))
        if name == 'dish':
            color = 'k'
            objects_list.append((position[0], position[1], name, color))
        if name == 'soup':
            # print("OBJ", obj)
            try:
                (soup_type, num_items, cook_time) = obj['state']
                obj_is_cooking = False
                obj_is_ready = False
                if num_items == 3:
                    if cook_time >= 20:
                        obj_is_ready = True
                    if cook_time < 20:
                        obj_is_cooking = True

                if obj_is_cooking is True:
                    color = 'r'
                elif obj_is_ready is True:
                    color = 'g'
                else:
                    color = 'orange'
            except:
                if obj['is_cooking'] is True:
                    color = 'r'
                elif obj['is_ready'] is True:
                    color = 'g'
                else:
                    color = 'orange'
            objects_list.append((position[0], position[1], name, color))

    return objects_list


def obj_p1(t, a, p1_data):
    #     t_partial = t[int(a)-1:int(a)+1]
    #     print('t_input', t)
    color = 'k'
    if p1_data[int(a)]['held_object'] is None:
        return [None, None, None, None]
    else:
        name = p1_data[int(a)]['held_object']['name']
        if name == 'dish':
            color = 'k'
        elif name == 'onion':
            color = 'y'
    #         elif name == 'onion':
    #             color = 'y'
    return p1_data[int(a)]['position'][0], p1_data[int(a)]['position'][1], p1_data[int(a)]['held_object']['name'], color


def obj_p2(t, a, p2_data):
    color = 'k'
    if p2_data[int(a)]['held_object'] is None:
        return [None, None, None, None]
    else:
        name = p2_data[int(a)]['held_object']['name']
        if name == 'dish':
            color = 'k'
        elif name == 'onion':
            color = 'y'
    return p2_data[int(a)]['position'][0], p2_data[int(a)]['position'][1], p2_data[int(a)]['held_object']['name'], color




def action_p1(t, a, p1_data):
    if p1_data[int(a)] == 'interact' or p1_data[int(a)] == 'INTERACT':
        act = 'I'
    else:
        x, y = p1_data[int(a)][0], p1_data[int(a)][1]
        act = 'N'
        if (x, y) == NORTH:
            act = 'N'
        if (x, y) == SOUTH:
            act = 'S'
        if (x, y) == EAST:
            act = 'E'
        if (x, y) == WEST:
            act = 'W'
        if (x, y) == STAY:
            act = 'Y'

    return act


def action_p2(t, a, p2_data):
    if p2_data[int(a)] == 'interact' or p2_data[int(a)] == 'INTERACT':
        act = 'I'
    else:
        x, y = p2_data[int(a)][0], p2_data[int(a)][1]
        act = 'N'
        if (x, y) == NORTH:
            act = 'N'
        if (x, y) == SOUTH:
            act = 'S'
        if (x, y) == EAST:
            act = 'E'
        if (x, y) == WEST:
            act = 'W'
        if (x, y) == STAY:
            act = 'Y'

    return act


def compute_steps_per_object_transfer(old_trials, p1_data, p2_data, objects_data, p1_actions,
                                      p2_actions, name, time_elapsed):
    N_steps = len(p1_data)
    a_min = 1  # the minimial value of the paramater a
    a_max = N_steps - 1  # the maximal value of the paramater a
    a_init = 1  # the value of the parameter a to be used initially, when the graph is created

    t = np.linspace(0, N_steps - 1, N_steps)

    counter_location_to_id = {
        (2, 1): 1,
        (2, 2): 2,
        (2, 3): 3,
        (1, 0): 4,
        (1, 4): 5,
        (4, 2): 6,
        (4, 3): 7
    }

    obj_count_id = 0
    next_obj_count_id = 0

    object_list_tracker = {}
    object_location_tracker = {}

    ordered_delivered_tracker = {}
    top_soup_counter_id = 0  # (3,0)
    right_soup_counter_id = 0  # (4,1)
    top_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (3,0)
    right_soup_contents_dict = {'this_contents': [], 'other_contents': [], 'other_state': 'empty'}  # (4,1)
    # states: empty, cooking, cooked, partial

    p2_carrying_soup = None
    p2_carrying_soup_pot_side = None
    p2_carrying_soup_pot_side_id = None
    absolute_order_counter = 0
    p2_time_picked_up_soup = None
    p2_time_delivered_soup = None

    players_holding = {1: None, 2: None}

    layout = eval(old_trials[old_trials['layout_name'] == name]['layout'].to_numpy()[0])
    layout = np.array([list(elem) for elem in layout])
    grid_display = np.zeros((layout.shape[0], layout.shape[1], 3))

    # loop over your images
    for a in range(len(t) - 1):
        # for a in range(100):

        p1_x, p1_y = f_p1(t, a, p1_data)[0], f_p1(t, a, p1_data)[1]
        p1_dir_x, p1_dir_y = arrow_p1(t, a, p1_data)[2], arrow_p1(t, a, p1_data)[3]
        p1_obj_x, p1_obj_y, p1_obj_name = obj_p1(t, a, p1_data)[0], \
                                          obj_p1(t, a, p1_data)[1], \
                                          obj_p1(t, a, p1_data)[2]
        p1_act = action_p1(t, a, p1_actions)

        p2_x, p2_y = f_p2(t, a, p1_data)[0], f_p2(t, a, p1_data)[1]
        p2_dir_x, p2_dir_y = arrow_p2(t, a, p2_data)[2], arrow_p2(t, a, p2_data)[3]
        p2_obj_x, p2_obj_y, p2_obj_name = obj_p2(t, a, p2_data)[0], \
                                          obj_p2(t, a, p2_data)[1], \
                                          obj_p2(t, a, p2_data)[2]
        p2_act = action_p2(t, a, p2_actions)
        objects_list = world_obj(t, a, objects_data)
        t1 = time_elapsed[a]

        # cook_state
        # 0 = uncooked
        # 1 = cooking
        # 2 = cooked

        b = a + 1
        p1_x_next, p1_y_next = f_p1(t, b, p1_data)[0], f_p1(t, b, p1_data)[1]
        p1_dir_x_next, p1_dir_y_next = arrow_p1(t, b, p1_data)[2], arrow_p1(t, b, p1_data)[3]
        p1_obj_x_next, p1_obj_y_next, p1_obj_name_next = obj_p1(t, b, p1_data)[0], \
                                                         obj_p1(t, b, p1_data)[1], \
                                                         obj_p1(t, b, p1_data)[2]
        p1_act_next = action_p1(t, b, p1_actions)

        p2_x_next, p2_y_next = f_p2(t, b, p2_data)[0], f_p2(t, b, p2_data)[1]
        p2_dir_x_next, p2_dir_y_next = arrow_p2(t, b, p2_data)[2], arrow_p2(t, b, p2_data)[3]
        p2_obj_x_next, p2_obj_y_next, p2_obj_name_next = obj_p2(t, b, p2_data)[0], \
                                                         obj_p2(t, b, p2_data)[1], \
                                                         obj_p2(t, b, p2_data)[2]
        p2_act_next = action_p2(t, a, p2_actions)
        objects_list_next = world_obj(t, b, objects_data)
        t2 = time_elapsed[b]

        ################## PLAYER 1'S MOVEMENT ##################
        # If P1 moves or stays
        if p1_act in ['N', 'S', 'E', 'W']:
            #         print(p1_act, (p1_dir_x_next, p1_dir_y_next))

            # If P1 is carrying something and moving (dish or onion)
            if players_holding[1] is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_held_id = players_holding[1]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

        # If P1 interacted
        if p1_act == 'I':
            # If P1 picked up an object
            if p1_obj_x is None and p1_obj_x_next is not None:
                obj_location = (p1_obj_x_next, p1_obj_y_next)
                object_location_tracker[obj_location] = obj_count_id

                if obj_count_id not in object_list_tracker:
                    object_list_tracker[obj_count_id] = {}
                object_list_tracker[obj_count_id]['name'] = p1_obj_name_next
                object_list_tracker[obj_count_id]['player_holding'] = 1
                object_list_tracker[obj_count_id]['id'] = obj_count_id
                object_list_tracker[obj_count_id]['n_actions_since_pickup'] = 0
                object_list_tracker[obj_count_id]['location'] = obj_location
                object_list_tracker[obj_count_id]['on_screen'] = True
                object_list_tracker[obj_count_id]['player_holding_list'] = [1]
                object_list_tracker[obj_count_id]['p1_n_actions_since_pickup'] = 1
                object_list_tracker[obj_count_id]['p2_n_actions_since_pickup'] = 0
                object_list_tracker[obj_count_id]['counter_used'] = []
                object_list_tracker[obj_count_id]['p1_time_started'] = t1
                object_list_tracker[obj_count_id]['p1_time_completed'] = None
                object_list_tracker[obj_count_id]['p2_time_started'] = None
                object_list_tracker[obj_count_id]['p2_time_completed'] = None

                players_holding[1] = obj_count_id

                obj_count_id += 1

            # If P1 put down an object
            if p1_obj_x is not None and p1_obj_x_next is None:
                object_held_id = players_holding[1]

                object_list_tracker[object_held_id]['player_holding'] = 0
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['player_holding_list'].append(0)
                object_list_tracker[object_held_id]['p1_n_actions_since_pickup'] += 1

                placed_obj_x, placed_obj_y = list(p1_x_next)[0] + p1_dir_x_next, list(p1_y_next)[0] + p1_dir_y_next
                old_obj_location = object_list_tracker[object_held_id]['location']
                new_obj_location = (placed_obj_x, placed_obj_y)
                object_list_tracker[object_held_id]['location'] = new_obj_location

                object_location_tracker.pop(old_obj_location, None)
                object_location_tracker[new_obj_location] = object_held_id

                counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                object_list_tracker[object_held_id]['p1_time_completed'] = t2

                players_holding[1] = None

        ################## PLAYER 2'S MOVEMENT ##################
        # If P2 moves or stays
        if p2_act in ['N', 'S', 'E', 'W']:

            # If P2 is moving with an object: onion, dish, or soup
            if players_holding[2] is not None:
                obj_location = (p2_obj_x_next, p2_obj_y_next)
                object_held_id = players_holding[2]

                prev_location = object_list_tracker[object_held_id]['location']
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['location'] = obj_location
                object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                object_location_tracker.pop(prev_location, None)
                object_location_tracker[obj_location] = object_held_id

        # If P2 interacted
        if p2_act == 'I':
            placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
            # If P2 picked up soup from the pot. P2 would be carrying a dish
            if p2_obj_name_next == 'soup':

                # P2 was already carrying this soup
                #                 if p2_obj_name == 'soup':
                #                     print('WHAT IS GOING ON', (p2_data[a], p2_actions[a]))
                #                     print('WHAT IS GOING ON NEXT', (p2_data[b], p2_actions[b]))
                #                     print('picked up from', (placed_x, placed_y))
                #                     print()

                # P2 filled up a dish with the soup
                if p2_obj_name == 'dish':
                    objects_status = objects_data[a]
                    objects_status_next = objects_data[b]
                    top_pot_status = 'empty'
                    right_pot_status = 'empty'
                    for obj_status in objects_status:
                        if obj_status['name'] == 'soup' and obj_status['position'] == [3, 0]:
                            if obj_status['is_cooking'] == True:
                                top_pot_status = 'cooking'
                            if obj_status['is_ready'] == True:
                                top_pot_status = 'ready'
                            if obj_status['is_idle'] == True:
                                top_pot_status = 'idle'
                        if obj_status['name'] == 'soup' and obj_status['position'] == [4, 1]:
                            if obj_status['is_cooking'] == True:
                                right_pot_status = 'cooking'
                            if obj_status['is_ready'] == True:
                                right_pot_status = 'ready'
                            if obj_status['is_idle'] == True:
                                right_pot_status = 'idle'

                    top_soup_contents_dict['other_state'] = right_pot_status
                    right_soup_contents_dict['other_state'] = top_pot_status

                    p2_time_picked_up_soup = t1

                    # picked up from top counter
                    if (placed_x, placed_y) == (3, 0):
                        p2_carrying_soup = copy.deepcopy(top_soup_contents_dict)
                        p2_carrying_soup_pot_side = 'top'
                        p2_carrying_soup_pot_side_id = top_soup_counter_id

                        top_soup_contents_dict['this_contents'] = []  # (3,0)
                        right_soup_contents_dict['other_contents'] = []
                        right_soup_contents_dict['other_state'] = 'empty'
                        top_soup_counter_id += 1

                    #                     objects_data_next [{'name': 'soup', 'position': [3, 0], '_ingredients': [{'name': 'onion', 'position': [3, 0]}], 'cooking_tick': -1, 'is_cooking': False, 'is_ready': False, 'is_idle': True, 'cook_time': -1, '_cooking_tick': -1}]

                    # picked up from right counter
                    if (placed_x, placed_y) == (4, 1):
                        p2_carrying_soup = copy.deepcopy(right_soup_contents_dict)
                        p2_carrying_soup_pot_side = 'right'
                        p2_carrying_soup_pot_side_id = right_soup_counter_id

                        right_soup_contents_dict['this_contents'] = []  # (4,1)
                        top_soup_contents_dict['other_contents'] = []
                        top_soup_contents_dict['other_state'] = 'empty'
                        right_soup_counter_id += 1

            # If P2 picked up an object
            if p2_obj_x is None and p2_obj_x_next is not None:
                placed_x, placed_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next

                if (placed_x, placed_y) in object_location_tracker:
                    obj_picked_id = object_location_tracker[(placed_x, placed_y)]
                else:
                    print('!!! problem p2 pickup not found')
                    nearest_key = min(list(object_location_tracker.keys()),
                                      key=lambda c: (c[0] - placed_x) ** 2 + (c[1] - placed_y) ** 2)
                    obj_picked_id = object_location_tracker[nearest_key]

                new_obj_location = (p2_obj_x_next, p2_obj_y_next)

                object_list_tracker[obj_picked_id]['player_holding'] = 2
                object_list_tracker[obj_picked_id]['n_actions_since_pickup'] += 1
                object_list_tracker[obj_picked_id]['location'] = new_obj_location
                object_list_tracker[obj_picked_id]['on_screen'] = True
                object_list_tracker[obj_picked_id]['player_holding_list'].append(2)
                object_list_tracker[obj_picked_id]['p2_n_actions_since_pickup'] += 1
                object_list_tracker[obj_picked_id]['p2_time_started'] = t1

                players_holding[2] = obj_picked_id

                object_location_tracker.pop((placed_x, placed_y), None)
                object_location_tracker[new_obj_location] = obj_picked_id

        # If P2 put down an object
        if p2_obj_x is not None and p2_obj_x_next is None:
            object_held_id = players_holding[2]
            placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next

            # If P2 delivered a soup
            if p2_obj_name == 'soup' and (placed_obj_x, placed_obj_y) == (3, 4):
                p2_time_delivered_soup = t2
                ordered_delivered_tracker[absolute_order_counter] = {}
                ordered_delivered_tracker[absolute_order_counter]['details'] = p2_carrying_soup
                ordered_delivered_tracker[absolute_order_counter]['pot_side'] = p2_carrying_soup_pot_side
                ordered_delivered_tracker[absolute_order_counter]['pot_side_id'] = p2_carrying_soup_pot_side_id
                ordered_delivered_tracker[absolute_order_counter]['time_picked_up'] = p2_time_picked_up_soup
                ordered_delivered_tracker[absolute_order_counter]['time_delivered'] = p2_time_delivered_soup

                absolute_order_counter += 1




            else:
                # placed at top counter pot
                if (placed_obj_x, placed_obj_y) == (3, 0):
                    top_soup_contents_dict['this_contents'].append(object_held_id)
                    right_soup_contents_dict['other_contents'].append(object_held_id)

                # placed at right counter pot
                if (placed_obj_x, placed_obj_y) == (4, 1):
                    right_soup_contents_dict['this_contents'].append(object_held_id)
                    top_soup_contents_dict['other_contents'].append(object_held_id)

                object_list_tracker[object_held_id]['player_holding'] = 0
                object_list_tracker[object_held_id]['n_actions_since_pickup'] += 1
                object_list_tracker[object_held_id]['player_holding_list'].append(0)
                object_list_tracker[object_held_id]['p2_n_actions_since_pickup'] += 1

                placed_obj_x, placed_obj_y = list(p2_x_next)[0] + p2_dir_x_next, list(p2_y_next)[0] + p2_dir_y_next
                old_obj_location = object_list_tracker[object_held_id]['location']
                new_obj_location = (placed_obj_x, placed_obj_y)
                object_list_tracker[object_held_id]['location'] = new_obj_location

                object_location_tracker.pop(old_obj_location, None)
                object_location_tracker[new_obj_location] = object_held_id

                if (placed_obj_x, placed_obj_y) in counter_location_to_id:
                    counter_index = counter_location_to_id[(placed_obj_x, placed_obj_y)]
                    object_list_tracker[object_held_id]['counter_used'].append(counter_index)

                object_list_tracker[object_held_id]['p2_time_completed'] = t2

            players_holding[2] = None

    return object_list_tracker, ordered_delivered_tracker


def pull_features_from_output(object_list_tracker, ordered_delivered_tracker):
    other_pot_contains_num_onions = []
    other_pot_states = []
    steps_p1_took_total_order = []
    steps_p2_took_total_order = []
    order_completion_times = []
    for order_id in ordered_delivered_tracker:
        n_other = len(ordered_delivered_tracker[order_id]['details']['other_contents'])
        other_pot_contains_num_onions.append(n_other)

        state_other = ordered_delivered_tracker[order_id]['details']['other_state']
        other_pot_states.append(state_other)

        onion_indices = ordered_delivered_tracker[order_id]['details']['this_contents']
        steps_p1_took = 0
        steps_p2_took = 0
        for onion_idx in onion_indices:
            steps_p1_took += object_list_tracker[onion_idx]['p1_n_actions_since_pickup']
            steps_p2_took += object_list_tracker[onion_idx]['p2_n_actions_since_pickup']
        steps_p1_took_total_order.append(steps_p1_took)
        steps_p2_took_total_order.append(steps_p2_took)

        order_start_time = object_list_tracker[onion_idx]['p1_time_started']
        order_end_time = ordered_delivered_tracker[order_id]['time_delivered']
        time_to_complete = order_end_time - order_start_time
        order_completion_times.append(time_to_complete)

    return other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times


def plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order,
                 order_completion_times):
    plt.scatter(range(len(other_pot_contains_num_onions)), other_pot_contains_num_onions)
    plt.plot(range(len(other_pot_contains_num_onions)), other_pot_contains_num_onions)
    plt.xlabel('Order Number')
    plt.ylabel("Number of Onions in Other Pot")
    plt.title("Number of Onions in Other Pot vs. Order Number")
    plt.show()
    plt.close()

    plt.scatter(range(len(other_pot_states)), other_pot_states)
    plt.plot(range(len(other_pot_states)), other_pot_states)
    plt.xlabel('Order Number')
    plt.ylabel("Other Pot State")
    plt.title("Other Pot State vs. Order Number")
    plt.show()
    plt.close()

    plt.scatter(range(len(steps_p1_took_total_order)), steps_p1_took_total_order)
    plt.scatter(range(len(steps_p2_took_total_order)), steps_p2_took_total_order)
    plt.xlabel('Order Number')
    plt.ylabel("N Steps Players Took")
    plt.title("Player Steps vs. Order Number")
    plt.legend(['P1', 'P2'])
    plt.plot(range(len(steps_p1_took_total_order)), steps_p1_took_total_order)
    plt.plot(range(len(steps_p2_took_total_order)), steps_p2_took_total_order)
    plt.show()
    plt.close()

    plt.scatter(range(len(order_completion_times)), order_completion_times)
    plt.plot(range(len(order_completion_times)), order_completion_times)
    plt.xlabel('Order Number')
    plt.ylabel("Order Completion Time")
    plt.title("Completion Time vs. Order Number")
    plt.show()
    plt.close()


def generate_data_from_order_features(orders_features_dict):
    X_data = []
    for key_name in orders_features_dict:
        key_data = orders_features_dict[key_name]
        if key_name == 'other_pot_states':
            continue

        X_data.append(key_data)

    X_data = np.array(X_data).T
    #     print(X_data.shape)
    return X_data


def run_feature_extraction():
    name = 'random0'
    title = 'Forced Coord'

    # name = 'random0'
    # title = 'Forced Coordination'
    old_trials = import_2019_data()
    layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()
    # name = 'random0'
    # title = 'Forced Coordination'
    trial_data = {}

    team_order_features_dict = {}
    team_num_to_score = {}

    for j in range(len(layout_trials)):
        orders_features_dict = {
            'other_pot_contains_num_onions': [],
            'other_pot_states': [],
            'steps_p1_took_total_order': [],
            'steps_p2_took_total_order': [],
            'order_completion_times': []
        }
        trial_id = layout_trials[j]
        #     print('trial_id', trial_id)
        trial_df = old_trials[old_trials['trial_id'] == trial_id]
        score = old_trials[old_trials['trial_id'] == trial_id]['score'].to_numpy()[-1]
        state_data = trial_df['state'].to_numpy()
        joint_actions = trial_df['joint_action'].to_numpy()
        time_elapsed = trial_df['time_elapsed'].to_numpy()

        p1_data = []
        p2_data = []
        p1_actions = []
        p2_actions = []
        state_data_eval = []
        objects_data = []
        for i in range(1, len(state_data)):
            prev_state_x = json_eval(state_data[i - 1])
            state_x = json_eval(state_data[i])
            joint_actions_i = literal_eval(joint_actions[i])
            p1_index = 1
            p2_index = 0

            p1_data.append(state_x['players'][p1_index])
            p2_data.append(state_x['players'][p2_index])
            state_data_eval.append(state_x)
            objects_data.append(state_x['objects'])

            p1_actions.append(joint_actions_i[p1_index])
            p2_actions.append(joint_actions_i[p2_index])

        object_list_tracker, ordered_delivered_tracker = compute_steps_per_object_transfer(old_trials, p1_data, p2_data,
                                                                                           objects_data, p1_actions,
                                                                                           p2_actions, name,
                                                                                           time_elapsed)

        other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, \
        steps_p2_took_total_order, order_completion_times = pull_features_from_output(object_list_tracker,
                                                                                      ordered_delivered_tracker)

        orders_features_dict['other_pot_contains_num_onions'].extend(other_pot_contains_num_onions)
        orders_features_dict['other_pot_states'].extend(other_pot_states)
        orders_features_dict['steps_p1_took_total_order'].extend(steps_p1_took_total_order)
        orders_features_dict['steps_p2_took_total_order'].extend(steps_p2_took_total_order)
        orders_features_dict['order_completion_times'].extend(order_completion_times)

        team_order_features_dict[trial_id] = orders_features_dict
        team_num_to_score[trial_id] = score
    #     plot_results(other_pot_contains_num_onions, other_pot_states, steps_p1_took_total_order, steps_p2_took_total_order, order_completion_times)
    #     print()d
    return team_order_features_dict, team_num_to_score


def cluster_features():
    # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    team_order_features_dict, _ = run_feature_extraction()

    team_order_clusters = {}

    for team_num in team_order_features_dict:
        orders_features_dict = team_order_features_dict[team_num]
        team_data = generate_data_from_order_features(orders_features_dict)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(team_data)
        kmeans_prediction = kmeans.predict(team_data)
        team_order_clusters[team_num] = kmeans_prediction

    #     plt.scatter(range(len(kmeans_prediction)), kmeans_prediction)
    #     plt.plot(range(len(kmeans_prediction)), kmeans_prediction)
    #     plt.title('Team Number: '+str(team_num))
    #     plt.show()

    return

if __name__ == '__main__':
    cluster_features()





