import copy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

from overcooked_ai_py.utils import save_pickle, load_pickle
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import save_pickle

from human_aware_rl.utils import reset_tf, set_global_seed, prepare_nested_default_dict_for_pickle, common_keys_equal
from human_aware_rl.ppo.ppo import get_ppo_agent, plot_ppo_run, PPO_DATA_DIR
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, OvercookedState
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
import tqdm, gym
from overcooked_ai_py.utils import mean_and_std_err
from overcooked_ai_py.mdp.actions import Direction, Action
from human_proxy_irl_agent import IRL_Agent

DEFAULT_ENV_PARAMS = {
    "horizon": 400
}

MAX_HORIZON = 1e10

class IRL_AgentSimulator(object):
    """
    Class used to get rollouts and evaluate performance of various types of agents.
    """

    def __init__(self, mdp_params, env_params={},  mdp_fn_params=None, force_compute=False):
        """
        mdp_params (dict): params for creation of an OvercookedGridworld instance through the `from_layout_name` method
        env_params (dict): params for creation of an OvercookedEnv
        mdp_fn_params (dict): params to setup random MDP generation
        force_compute (bool): whether should re-compute MediumLevelPlanner although matching file is found
        mlp_params (dict): params for MediumLevelPlanner
        """
        assert type(mdp_params) is dict, "mdp_params must be a dictionary"

        if mdp_fn_params is None:
            self.variable_mdp = False
            self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**mdp_params)
        else:
            self.variable_mdp = True
            self.mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params, **mdp_fn_params)

        self.env = SimulateIRL_OvercookedEnv(self.mdp_fn, **env_params)
        self.force_compute = force_compute

    def evaluate_agent_pair(self, agent_pair, partner_index, num_games=1, display=False, info=True):
        return self.env.get_rollouts(agent_pair, num_games, partner_index, display=display, info=info)


class IRL_PPO_AgentPair(object):
    """
    AgentPair is the N=2 case of AgentGroup. Unlike AgentGroup,
    it supports having both agents being the same instance of Agent.

    NOTE: Allowing duplicate agents (using the same instance of an agent
    for both fields can lead to problems if the agents have state / history)
    """

    def __init__(self, partner_agent, irl_agent, partner_index, partner_is_ppo=True):

        self.partner_agent = partner_agent
        self.partner_agent.set_agent_index(partner_index)
        self.partner_index = partner_index
        self.partner_is_ppo = partner_is_ppo

        self.irl_agent = irl_agent
        self.irl_agent.set_agent_index(1-partner_index)

        self.stochastic = False # Change to false if using action probabilities instead


    def joint_action(self, state):

        # partner_action_idx = np.argmax(self.partner_agent.action(state))
        # partner_action = Action.INDEX_TO_ACTION[partner_action_idx]


        # print("self.partner_agent.action(state)", self.partner_agent.action(state))
        if self.stochastic:
            partner_action_idx = np.random.choice(len(Action.ALL_ACTIONS), p=self.partner_agent.action(state))
            partner_action = Action.INDEX_TO_ACTION[partner_action_idx]
        else:
            partner_action = self.partner_agent.action(state)
            partner_action_idx = Action.ACTION_TO_INDEX[partner_action]

        irl_action = self.irl_agent.action(state) # TODO: Change to IRL agent decisions


        # output = (partner_action, irl_action)
        if self.partner_index == 0:
            output = (partner_action, irl_action)
        else:
            output = (irl_action, partner_action)

        return output

    def set_mdp(self, mdp):
        self.partner_agent.set_mdp(mdp)
        self.irl_agent.set_mdp(mdp)


    def reset(self):
        self.partner_agent.reset()
        self.irl_agent.reset()







class SimulateIRL_OvercookedEnv(object):
    """An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state_fn=None, horizon=MAX_HORIZON, debug=False):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")

        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.reset()
        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")

    def step(self, joint_action):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        next_state, sparse_reward, reward_shaping = self.mdp.get_state_transition(self.state, joint_action)
        self.cumulative_sparse_rewards += sparse_reward
        self.cumulative_shaped_rewards += reward_shaping
        self.state = next_state
        self.t += 1
        done = self.is_done()
        info = {'shaped_r': reward_shaping}
        if done:
            info['episode'] = {
                'ep_sparse_r': self.cumulative_sparse_rewards,
                'ep_shaped_r': self.cumulative_shaped_rewards,
                'ep_length': self.t
            }
        return (next_state, sparse_reward, done, info)

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.mdp = self.mdp_generator_fn()
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

    def is_done(self):
        """Whether the episode is over."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)


    def run_agents(self, agent_pair, partner_idx, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t).
        """
        assert self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0, \
            "Did not reset environment before running agents"
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))


        assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True))

        return np.array(trajectory), self.t, self.cumulative_sparse_rewards, self.cumulative_shaped_rewards

    def get_rollouts(self, agent_pair, num_games, partner_idx, display=False, final_state=False, agent_idx=0, reward_shaping=0.0,
                     display_until=np.Inf, info=True):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took),
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format
        (baselines, stable_baselines, etc)

        NOTE: standard trajectories format used throughout the codebase
        """
        # agent_idx=0
        # print(f'\n\n\n\n\nAGENT INDEX INPUT {agent_idx} \n\n\n\n\n\n')
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [],  # Individual dense (= sparse + shaped * rew_shaping) reward values
            "ep_dones": [],  # Individual done values

            # With shape (n_episodes, ):
            "ep_returns": [],  # Sum of dense and sparse rewards across each episode
            "ep_returns_sparse": [],  # Sum of sparse rewards across each episode
            "ep_lengths": [],  # Lengths of each episode
            "mdp_params": [],  # Custom MDP params to for each episode
            "env_params": [],  # Custom Env params for each episode
            "strat_weights": [], # strategy weights for beliefs
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(agent_pair, partner_idx, display=display,
                                                                                       include_final_state=final_state,
                                                                                       display_until=display_until)
            obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]
            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            # trajectories["env_params"].append(self.env_params)

            self.reset()
            agent_pair.reset()

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        if info: print("Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
            mu, np.std(trajectories["ep_returns"]), se, num_games, np.mean(trajectories["ep_lengths"]))
        )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        return trajectories



def load_baseline_models(layout):
    baseline_model_paths = {

        "random3": "../human_aware_rl/experiments/data/ppo_runs/ppo_bc_train_random3_BASELINE",
        "simple": "../human_aware_rl/experiments/data/ppo_runs/ppo_bc_train_simple_BASELINE",
        "random0": "../human_aware_rl/experiments/data/ppo_runs/ppo_bc_train_random0_BASELINE",
        "random1": "../human_aware_rl/experiments/data/ppo_runs/ppo_bc_train_random1_BASELINE",
        "unident": "../human_aware_rl/experiments/data/ppo_runs/ppo_bc_train_unident_s_BASELINE",

    }
    model_file = baseline_model_paths[layout]

    baseline_seed = 9456
    baseline_ppo_agent, baseline_ppo_config = get_ppo_agent(model_file, baseline_seed, best=False)

    baseline_seed = 5578
    baseline_ppo_agent2, baseline_ppo_config2 = get_ppo_agent(model_file, baseline_seed, best=False)

    return baseline_ppo_agent, baseline_ppo_config, baseline_ppo_agent2, baseline_ppo_config2



def run_sim(bc_agent, bc_params, baseline_ppo_agent, irl_agent, num_rounds=1, partner_is_ppo=True, partner_index=1):
    evaluator = IRL_AgentSimulator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])
    ppo_and_ppo = evaluator.evaluate_agent_pair(IRL_PPO_AgentPair(baseline_ppo_agent, irl_agent, partner_index=partner_index, partner_is_ppo=partner_is_ppo),
                                                partner_index=partner_index, num_games=num_rounds, display=False)
    # rewards = ppo_and_ppo["ep_rewards"][0]
    avg_ppo_and_ppo_rewards = np.mean(ppo_and_ppo['ep_returns'])

    return avg_ppo_and_ppo_rewards


def load_bc_partner(layout):
    bc_model_paths = {


        "random1": "../human_aware_rl/experiments/data/bc_models/random1_bc_test_seed0",

    }
    # print("bc_model_paths[layout]", bc_model_paths[layout])
    bc_agent, bc_params = get_bc_agent_from_saved(bc_model_paths[layout])

    return bc_agent, bc_params


def simulate_belief_update(layout):
    bc_agent, bc_params = load_bc_partner(layout)

    irl_agent = IRL_Agent(layout)

    baseline_ppo_agent, baseline_ppo_config, baseline_ppo_agent2, baseline_ppo_config2 = load_baseline_models(layout)

    rewards = run_sim(bc_agent, bc_params, baseline_ppo_agent, irl_agent, num_rounds=1, partner_is_ppo=True, partner_index=1)

    print("rewards", rewards)





if __name__ == "__main__":
    layout = 'random1'
    simulate_belief_update(layout)







