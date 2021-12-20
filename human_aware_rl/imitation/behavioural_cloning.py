import gym
import tqdm, copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset

# import sys
# sys.path.insert(0, "../../")

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import save_pickle, load_pickle

from human_aware_rl.utils import reset_tf, set_global_seed, common_keys_equal
from human_aware_rl.baselines_utils import create_dir_if_not_exists
from human_aware_rl.human.process_dataframes import save_npz_file, get_trajs_from_data, get_trajs_from_data_selective, \
    get_trajs_from_data_for_cross_validation, get_trajs_from_data_specify_groups

BC_SAVE_DIR = "../data/bc_runs/"

# DEFAULT_DATA_PARAMS = {
#     "train_mdps": ["simple"],
#     "ordered_trajs": True,
#     "human_ai_trajs": False,
#     "data_path": "../data/human/anonymized/clean_train_trials.pkl"
# }
DEFAULT_DATA_PARAMS = {
    "train_mdps": ["unident_s"],
    "ordered_trajs": True,
    "human_ai_trajs": False,
    "data_path": "../data/human/anonymized/clean_main_trials.pkl"
}

DEFAULT_BC_PARAMS = {
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {}, # Nothing to overwrite defaults
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {}
}

def init_gym_env(bc_params):
    env_setup_params = copy.deepcopy(bc_params)
    del env_setup_params["data_params"] # Not necessary for setting up env
    mdp = OvercookedGridworld.from_layout_name(**bc_params["mdp_params"])
    env = OvercookedEnv(mdp, **bc_params["env_params"])
    gym_env = gym.make("Overcooked-v0")
    
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
    gym_env.custom_init(env, featurize_fn=lambda x: mdp.featurize_state(x, mlp))
    return gym_env

def train_bc_agent(model_save_dir, bc_params, num_epochs=1000, lr=1e-4, adam_eps=1e-8):
    # Extract necessary expert data and save in right format
    set_global_seed(64)
    expert_trajs = get_trajs_from_data(**bc_params["data_params"])
    # Load the expert dataset
    save_npz_file(expert_trajs, "temp.npz")
    # Create a stable-baselines ExpertDataset
    dataset = ExpertDataset(expert_path="temp.npz", verbose=1, train_fraction=0.85)
    assert dataset is not None
    assert dataset.train_loader is not None


    # Pass the ExpertDataset into BC model and params
    # Return the BC model
    return bc_from_dataset_and_params(dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps)

def train_bc_agent_cross_validation(is_train, train_workers, test_workers, model_save_dir, bc_params, num_epochs=1000, lr=1e-4, adam_eps=1e-8):
    # Extract necessary expert data and save in right format
    set_global_seed(64)
    expert_trajs = get_trajs_from_data_for_cross_validation(is_train, train_workers, test_workers, **bc_params["data_params"])
    # Load the expert dataset
    save_npz_file(expert_trajs, "temp.npz")
    # Create a stable-baselines ExpertDataset
    dataset = ExpertDataset(expert_path="temp.npz", verbose=1, train_fraction=0.85)
    assert dataset is not None
    assert dataset.train_loader is not None


    # Pass the ExpertDataset into BC model and params
    # Return the BC model
    return bc_from_dataset_and_params(dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps)



def train_bc_agent_w_finetuning(selected_workers, model_save_dir, bc_params, num_epochs=1000, lr=1e-4, adam_eps=1e-8):
    # Random 3
    # {2: 13, 4: 23, 13: 68, 15: 78, 16: 83, 17: 88, 19: 98, 20: 103}
    # {1: 8, 3: 18, 10: 53, 11: 58, 12: 63, 18: 93, 22: 113}

    # train_workers = [1,3,10,11,12,18,22] # swap for random3
    # test_workers = [2,4,13,15,16,17,19,20]

    # AA
    # {2: 11, 3: 16, 11: 56, 12: 61, 13: 66, 14: 71, 16: 81, 20: 101, 22: 111}
    # {1: 6, 4: 21, 10: 51, 15: 76, 17: 86, 18: 91, 19: 96, 23: 116}
    # train_workers = [2, 15, 19, 4, 20, 3, 11, 12, 16]
    # test_workers = [1, 10, 17, 18, 23, 22, 14, 13]
    # Strat 0 - 2, 15, 19
    # Strat 1 - 4, 20
    # Strat 2 - 3, 11, 12
    # Strat 3 - 16

    # if 'train' in model_save_dir:
    #     is_train = True
    # else:
    #     is_train = False

    # Extract necessary expert data and save in right format
    set_global_seed(64)
    selective = False
    # is_train = True
    # expert_trajs = get_trajs_from_data_selective(selected_worker_ids, **bc_params["data_params"])
    expert_trajs = get_trajs_from_data_specify_groups(selective, selected_workers=None, **bc_params["data_params"])
    # Load the expert dataset
    save_npz_file(expert_trajs, "temp.npz")
    # Create a stable-baselines ExpertDataset
    dataset = ExpertDataset(expert_path="temp.npz", verbose=1, train_fraction=0.85)
    assert dataset is not None
    assert dataset.train_loader is not None

    ## GET DATASET FOR FINETUNING
    # Extract necessary expert data and save in right format
    selective = True
    # is_train = False
    # expert_trajs = get_trajs_from_data_selective(**bc_params["data_params"])
    expert_trajs = get_trajs_from_data_specify_groups(selective, selected_workers=selected_workers,
                                                      **bc_params["data_params"])
    # Load the expert dataset
    save_npz_file(expert_trajs, "temp_finetune.npz")
    # Create a stable-baselines ExpertDataset
    finetune_dataset = ExpertDataset(expert_path="temp_finetune.npz", verbose=1, train_fraction=0.85)
    assert finetune_dataset is not None
    assert finetune_dataset.train_loader is not None


    # Pass the ExpertDataset into BC model and params
    # Return the BC model
    return bc_w_finetune_from_dataset_and_params(dataset, finetune_dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps)




def bc_from_dataset_and_params(dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps):
    # Setup env
    gym_env = init_gym_env(bc_params)

    # Train and save model
    create_dir_if_not_exists(BC_SAVE_DIR + model_save_dir)

    # Create stable-baselines GAIL base model
    model = GAIL("MlpPolicy", gym_env, dataset, verbose=1)
    # Only pretrain the GAIL model, which is just supervised BC on the ExpertDataset.
    model.pretrain(dataset, n_epochs=num_epochs, learning_rate=lr, adam_epsilon=adam_eps, save_dir=BC_SAVE_DIR + model_save_dir)

    # Save BC Model
    save_bc_model(model_save_dir, model, bc_params)
    return model

def bc_w_finetune_from_dataset_and_params(dataset, finetune_dataset, bc_params, model_save_dir, num_epochs, lr, adam_eps):
    # Setup env
    gym_env = init_gym_env(bc_params)

    # Train and save model
    create_dir_if_not_exists(BC_SAVE_DIR + model_save_dir)

    # Create stable-baselines GAIL base model
    model = GAIL("MlpPolicy", gym_env, dataset, verbose=1)
    # Only pretrain the GAIL model, which is just supervised BC on the ExpertDataset.
    model.pretrain_and_finetune(dataset, finetune_dataset, n_epochs=num_epochs, learning_rate=lr, adam_epsilon=adam_eps, save_dir=BC_SAVE_DIR + model_save_dir)

    # Save BC Model
    save_bc_model(model_save_dir, model, bc_params)
    return model

def save_bc_model(model_save_dir, model, bc_params):
    print("Saved BC model at", BC_SAVE_DIR + model_save_dir)
    print(model_save_dir)
    model.save(BC_SAVE_DIR + model_save_dir + "model")
    bc_metadata = {
        "bc_params": bc_params,
        "train_info": model.bc_info
    }
    save_pickle(bc_metadata, BC_SAVE_DIR + model_save_dir + "bc_metadata")

def get_bc_agent_from_saved(model_name, no_waits=False):
    model, bc_params = load_bc_model_from_path(model_name)
    return get_bc_agent_from_model(model, bc_params, no_waits), bc_params

def get_bc_agent_from_model(model, bc_params, no_waits=False):
    mdp = OvercookedGridworld.from_layout_name(**bc_params["mdp_params"])
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
    
    def encoded_state_policy(observations, include_waits=True, stochastic=False):
        # Input observations are a list of featurized states.
        # Pass the observations of each state into the action probability function of the model.
        # Returns a list of action probabilities for each observation at every timestep.
        action_probs_n = model.action_probability(observations)

        # If include_waits is False, remove the indices and renormalize. Probably mix maximum action.
        if not include_waits:
            action_probs = ImitationAgentFromPolicy.remove_indices_and_renormalize(action_probs_n, [Action.ACTION_TO_INDEX[Direction.STAY]])

        # If include_waits is True, choose stochastically, could be suboptimal decisions made. Sample based on action probs.
        if stochastic:
            return [np.random.choice(len(action_probs[i]), p=action_probs[i]) for i in range(len(action_probs))]
        return action_probs_n

    def state_policy(mdp_states, agent_indices, include_waits, stochastic=False):
        # encode_fn = lambda s: mdp.preprocess_observation(s)

        # Take the set of mdp states. Featurize every mdp state, so that every state is a feature vector (observation).
        # Add it to a list of observations, turn to numpy array.
        # Pass set of featurized states to the encoded state policy function.

        encode_fn = lambda s: mdp.featurize_state(s, mlp)

        obs = []
        for agent_idx, s in zip(agent_indices, mdp_states):
            ob = encode_fn(s)[agent_idx]
            obs.append(ob)
        obs = np.array(obs)
        action_probs = encoded_state_policy(obs, include_waits, stochastic)
        return action_probs

    return ImitationAgentFromPolicy(state_policy, encoded_state_policy, no_waits=no_waits, mlp=mlp)

def eval_with_benchmarking_from_model(n_games, model, bc_params, no_waits, display=False):
    bc_params = copy.deepcopy(bc_params)
    # Both agents are the same model, same params
    a0 = get_bc_agent_from_model(model, bc_params, no_waits)
    a1 = get_bc_agent_from_model(model, bc_params, no_waits)
    del bc_params["data_params"], bc_params["mdp_fn_params"]
    a_eval = AgentEvaluator(**bc_params)
    ap = AgentPair(a0, a1)
    # Get rollouts from playing both agents (identical agents) with each other.
    trajectories = a_eval.evaluate_agent_pair(ap, num_games=n_games, display=display)
    return trajectories

def eval_with_benchmarking_from_saved(n_games, model_name, no_waits=False, display=False):
    model, bc_params = load_bc_model_from_path(model_name)
    return eval_with_benchmarking_from_model(n_games, model, bc_params, no_waits, display=display)

def load_bc_model_from_path(model_name):
    # NOTE: The lowest loss and highest accuracy models 
    # were also saved, can be found in the same dir with
    # special suffixes.
    bc_metadata = load_pickle(BC_SAVE_DIR + model_name + "/bc_metadata")
    bc_params = bc_metadata["bc_params"]
    model = GAIL.load(BC_SAVE_DIR + model_name + "/model")
    return model, bc_params

def plot_bc_run(run_info, num_epochs):
    xs = range(0, num_epochs, max(int(num_epochs/10), 1))
    plt.plot(xs, run_info['train_losses'], label="train loss")
    plt.plot(xs, run_info['val_losses'], label="val loss")
    plt.plot(xs, run_info['val_accuracies'], label="val accuracy")
    plt.legend()
    plt.show()

def plot_bc_run_modified(run_info, num_epochs, seed_idx, seed):
    xs = range(0, num_epochs, max(int(num_epochs / 10), 1))
    plt.plot(xs, run_info['train_losses'], label="train loss")
    plt.plot(xs, run_info['val_losses'], label="val loss")
    plt.plot(xs, run_info['val_accuracies'], label="val accuracy")
    plt.legend()
    plt.savefig('../../exploration_images/bc_run_seedidx'+str(seed_idx)+'_seed'+str(seed)+'.png')
    plt.close()


class ImitationAgentFromPolicy(AgentFromPolicy):
    """Behavior cloning agent interface"""
    #
    def __init__(self, state_policy, direct_policy, mlp=None, stochastic=True, no_waits=False, stuck_time=3):
        super().__init__(state_policy, direct_policy)
        # How many turns in same position to be considered 'stuck'
        self.stuck_time = stuck_time
        self.history_length = stuck_time + 1
        self.stochastic = stochastic

        self.action_probs = False
        self.no_waits = no_waits
        self.will_unblock_if_stuck = False if stuck_time == 0 else True
        self.mlp = mlp
        self.reset()

    def action(self, state):
        return self.actions(state)

    def actions(self, states, agent_indices=None):
        """
        The standard action function call, that takes in a Overcooked state
        and returns the corresponding action.

        Requires having set self.agent_index and self.mdp
        """
        if agent_indices is None:
            assert isinstance(states, OvercookedState)
            # Chose to overwrite agent index, set it as current agent index. Useful for Vectorized environments
            agent_indices = [self.agent_index]
            states = [states]
        
        # Actually now state is a list of states
        assert len(states) > 0

        all_actions = self.multi_action(states, agent_indices)

        if len(agent_indices) > 1:
            return all_actions
        return all_actions[0]

    def multi_action(self, states, agent_indices):
        try:
            action_probs_n = list(self.state_policy(states, agent_indices, not self.no_waits))
        except AttributeError:
            raise AttributeError("Need to set the agent_index or mdp of the Agent before using it")

        all_actions = []
        for parallel_agent_idx, curr_agent_action_probs in enumerate(action_probs_n):
            curr_agent_idx = agent_indices[parallel_agent_idx]
            curr_agent_state = states[parallel_agent_idx]
            self.set_agent_index(curr_agent_idx)
            
            # Removing wait action
            if self.no_waits:
                curr_agent_action_probs = self.remove_indices_and_renormalize(curr_agent_action_probs, [Action.ACTION_TO_INDEX[Direction.STAY]])

            if self.will_unblock_if_stuck:
                curr_agent_action_probs = self.unblock_if_stuck(curr_agent_state, curr_agent_action_probs)

            if self.stochastic:
                action_idx = np.random.choice(len(curr_agent_action_probs), p=curr_agent_action_probs)
            else:
                # curr_agent_action_probs /= sum(curr_agent_action_probs)
                # if np.sum(curr_agent_action_probs) != 0:
                #     curr_agent_action_probs /= np.sum(curr_agent_action_probs)
                # else:
                #     curr_agent_action_probs /= 0.0000001

                # print("\n\ncurr_agent_action_probs", curr_agent_action_probs)
                action_idx = np.argmax(curr_agent_action_probs)

                # max_action_prob = np.max(curr_agent_action_probs)
                # if max_action_prob < 0.5:
                #     action_idx = 4
                # print("selected action idx = ", action_idx)

            curr_agent_action = Action.INDEX_TO_ACTION[action_idx]
            self.add_to_history(curr_agent_state, curr_agent_action)

            if self.action_probs:
                all_actions.append(curr_agent_action_probs)
            else:
                all_actions.append(curr_agent_action)
        return all_actions

    def unblock_if_stuck(self, state, action_probs):
        """Get final action for a single state, given the action probabilities
        returned by the model and the current agent index.
        NOTE: works under the invariance assumption that self.agent_idx is already set
        correctly for the specific parallel agent we are computing unstuck for"""
        stuck, last_actions = self.is_stuck(state)
        if stuck:
            assert any([a not in last_actions for a in Direction.ALL_DIRECTIONS]), last_actions
            last_action_idxes = [Action.ACTION_TO_INDEX[a] for a in last_actions]
            action_probs = self.remove_indices_and_renormalize(action_probs, last_action_idxes)
        return action_probs

    def is_stuck(self, state):
        if None in self.history[self.agent_index]:
            return False, []
        
        last_states = [s_a[0] for s_a in self.history[self.agent_index][-self.stuck_time:]]
        last_actions = [s_a[1] for s_a in self.history[self.agent_index][-self.stuck_time:]]
        player_states = [s.players[self.agent_index] for s in last_states]
        pos_and_ors = [p.pos_and_or for p in player_states] + [state.players[self.agent_index].pos_and_or]
        if self.checkEqual(pos_and_ors):
            return True, last_actions
        return False, []

    @staticmethod
    def remove_indices_and_renormalize(probs, indices):
        # This will induce random lack of movement
        # If include_waits = False, run this function.
        # If probs is more than 1 dimensional: more than 1 action, or more than 1 timestep
        #
        if len(np.array(probs).shape) > 1:
            probs = np.array(probs)
            for row_idx, row in enumerate(indices):
                for idx in indices:
                    probs[row_idx][idx] = 0
            norm_probs =  probs.T / np.sum(probs, axis=1)
            return norm_probs.T
        else:
            for idx in indices:
                probs[idx] = 0
            return probs / sum(probs)

    def checkEqual(self, iterator):
        first_pos_and_or = iterator[0]
        for curr_pos_and_or in iterator:
            if curr_pos_and_or[0] != first_pos_and_or[0] or curr_pos_and_or[1] != first_pos_and_or[1]:
                return False
        return True

    def add_to_history(self, state, action):
        assert len(self.history[self.agent_index]) == self.history_length
        self.history[self.agent_index].append((state, action))
        self.history[self.agent_index] = self.history[self.agent_index][1:]

    def reset(self):
        # Matrix of histories, where each index/row corresponds to a specific agent
        self.history = defaultdict(lambda: [None] * self.history_length)






##########
# EXTRAS #
##########

def stable_baselines_predict_fn(model, observation):
    a_probs = model.action_probability(observation)
    action_idx = np.random.choice(len(a_probs), p=a_probs)
    return action_idx

def eval_with_standard_baselines(n_games, model_name, display=False):
    """Method to evaluate agent performance with stable-baselines infrastructure,
    just to make sure everything is compatible and integrating correctly."""
    bc_metadata = load_pickle(BC_SAVE_DIR + model_name + "/bc_metadata")
    bc_params = bc_metadata["bc_params"]
    model = GAIL.load(BC_SAVE_DIR + model_name + "/model")

    gym_env = init_gym_env(bc_params)

    tot_rew = 0
    for i in tqdm.trange(n_games):
        obs, _ = gym_env.reset()
        done = False
        while not done:
            ob0, ob1 = obs
            a0 = stable_baselines_predict_fn(model, ob0)
            a1 = stable_baselines_predict_fn(model, ob1)
            joint_action = (a0, a1)
            (obs, _), rewards, done, info = gym_env.step(joint_action)
            tot_rew += rewards

    print("avg reward", tot_rew / n_games)
    return tot_rew / n_games

def symmetric_bc(model_savename, bc_params, num_epochs=1000, lr=1e-4, adam_eps=1e-8):
    """DEPRECATED: Trains two BC models from the same data. Splits data 50-50 and uses each subset as training data for
    one model and validation for the other."""
    expert_trajs = get_trajs_from_data(bc_params["data_params"])

    save_npz_file(expert_trajs, "temp")
    train_dataset = ExpertDataset(expert_path="temp", verbose=1, train_fraction=0.5)
    train_indices = train_dataset.train_loader.original_indices
    val_indices = train_dataset.val_loader.original_indices

    # Train BC model
    train_model_save_dir = model_savename + "_train/"
    bc_from_dataset_and_params(train_dataset, bc_params, train_model_save_dir, num_epochs, lr, adam_eps)

    # Switching testing and validation datasets (somewhat hacky)
    indices_split = (val_indices, train_indices)
    test_dataset = ExpertDataset(expert_path="temp", verbose=1, train_fraction=0.5, indices_split=indices_split)

    # Test BC model
    test_model_save_dir = model_savename + "_test/"
    bc_from_dataset_and_params(test_dataset, bc_params, test_model_save_dir, num_epochs, lr, adam_eps)
