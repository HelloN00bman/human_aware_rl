from dependencies import *
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction

def json_eval(s):
    json_acceptable_string = s.replace("'", "\"")
    d = json.loads(json_acceptable_string)
    return d



def import_2019_data():
    hh_all_2019_file = '../human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle'

    with open(hh_all_2019_file,'rb') as file:
        humans_2019_file = pkl.load(file)

    # humans_2019_file.to_csv('humans_all_2019.csv')
    old_trials = humans_2019_file
    return old_trials




def run_featurization():
    name = 'random0'

    old_trials = import_2019_data()
    layout_trials = old_trials[old_trials['layout_name'] == name]['trial_id'].unique()

    overcooked_mdp = OvercookedGridworld.from_layout_name('random0', start_order_list=['any'], cook_time=20)
    base_params_start_or = {
        'start_orientations': True,
        'wait_allowed': False,
        'counter_goals': overcooked_mdp.terrain_pos_dict['X'],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': False
    }
    mlp = MediumLevelPlanner(overcooked_mdp, base_params_start_or)

    for j in range(len(layout_trials)):

        trial_id = layout_trials[j]
        trial_df = old_trials[old_trials['trial_id'] == trial_id]

        score = old_trials[old_trials['trial_id'] == trial_id]['score'].to_numpy()[-1]
        state_data = trial_df['state'].values
        joint_actions = trial_df['joint_action'].values
        time_elapsed = trial_df['time_elapsed'].values

        # oc_state = OvercookedState()
        player_idx = 0

        X = []
        Y = []

        for state_i in range(1, len(state_data)):
            # print("json_eval(state_data[state_i])", json_eval(state_data[state_i]))
            overcooked_state_i = OvercookedState.from_dict(json_eval(state_data[state_i]))
            # featurized_state_i = featurize_state(overcooked_state_i)
            prev_joint_action_eval = json_eval(joint_actions[state_i-1])
            prev_joint_action = []
            for elem in prev_joint_action_eval:
                if tuple(elem) !=  ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T'):
                    prev_joint_action.append(tuple(elem))
                else:
                    prev_joint_action.append("interact")


            joint_action_eval = json_eval(joint_actions[state_i])
            joint_action = []
            for elem in joint_action_eval:
                if tuple(elem) != ('I', 'N', 'T', 'E', 'R', 'A', 'C', 'T'):
                    joint_action.append(tuple(elem))
                else:
                    joint_action.append("interact")
            action_label = Action.ACTION_TO_INDEX[joint_action[player_idx]]
            action_label = np.eye(6)[action_label]

            # print("prev_joint_action", prev_joint_action)
            # print('overcooked_state_i', overcooked_state_i)
            ordered_features_p0, ordered_features_p1 = overcooked_mdp.featurize_state_complex(overcooked_state_i, mlp, prev_joint_action)
            # print("featurized_state", ordered_features_p0)
            X.append(ordered_features_p0)
            Y.append(action_label)
        break

    X = np.array(X)
    Y = np.array(Y)
    return X, Y




def test_behavior_cloning():
    X, Y = run_featurization()
    print("X shaoe", X.shape)
    print('Y', Y.shape)

    device = torch.device('cpu')
    # device = torch.device('cuda') # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 128, 116, 100, 6

    # Create random Tensors to hold inputs and outputs
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # After constructing the model we use the .to() method to move it to the
    # desired device.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ).to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function. Setting
    # reduction='sum' means that we are computing the *sum* of squared errors rather
    # than the mean; this is for consistency with the examples above where we
    # manually compute the loss, but in practice it is more common to use mean
    # squared error as a loss by setting reduction='elementwise_mean'.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    n_epochs = 2000
    for t in range(n_epochs):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the loss.
        # y_pred = np.argmax(y_pred, axis=0)
        # print('y_pred, y', (y_pred, y))

        loss = loss_fn(y_pred, y)
        # print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its data and gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad


    # TEST
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    y_pred = model(x).detach().numpy()
    y_true = Y
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    accuracy = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            accuracy += 1

    print("ACCURACY", accuracy/len(y_pred))



if __name__ == '__main__':
    test_behavior_cloning()












