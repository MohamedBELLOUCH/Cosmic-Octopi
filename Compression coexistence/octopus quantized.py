"""Quantized-uplink copy of Cosmic Octopi/octopus.py."""

# Load the shared codec from its requested filename containing a space.
from pathlib import Path
import importlib.util
import sys

_CORE = Path(__file__).resolve().parents[1] / "Cosmic Octopi"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
_CODEC_NAME = "_cosmic_octopi_quantized_transport"
if _CODEC_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _CODEC_NAME, Path(__file__).with_name("utils quantized.py"))
    _codec = importlib.util.module_from_spec(_spec)
    sys.modules[_CODEC_NAME] = _codec
    _spec.loader.exec_module(_codec)
_codec = sys.modules[_CODEC_NAME]
quantize_model = _codec.quantize_model
dequantize_model = _codec.dequantize_model
get_payload_size_in_kb = _codec.get_payload_size_in_kb

import numpy as np
import torch
from pymle.sim.Simulator1D import Simulator1D
from pymle.models import OrnsteinUhlenbeck
import gymnasium as gym
import importlib
import utils
# Reuse shared utilities without reloading their module state.
from utils import activations_candidacies
from utils import Network, Policy, TrajectorySampler, NormalizeAdvantages, EnvRunner, PPO, GAE
from utils import CLASS_NAMES, current_device
import copy

class Octopus:
    def __init__(self, agent_id, activation_periods, triggering_events_instants, number_of_episodes, 
                 global_iterations_instants, Xi_matrix, hyper_params_,
                 planet_gravity, environment_ids=None):
        if environment_ids is None:
            environment_ids = {
                'cheetah': 'HalfCheetah-v5',
                'ant': 'Ant-v5',
                'hopper': 'Hopper-v5',
                'humanoid': 'Humanoid-v5',
            }
        self.agent_id = agent_id
        self.activation_periods = activation_periods
        self.triggering_events_instants = triggering_events_instants
        self.n_classes = len(self.triggering_events_instants) if triggering_events_instants != None else 0
        self.number_of_episodes = number_of_episodes
        self.temperature = hyper_params_['temperature']
        self.scaling_constant = hyper_params_['scaling_constant']
        # model-free approach
        self.models = None
        self.local_environments = None

        # model-based approach
        self.Xi_matrix = Xi_matrix
        self.rewards_history = [[] for _ in range(self.n_classes)]
        self.last_global_iteration_indices = np.zeros(self.n_classes) # number of classes
        self.last_global_fitness_values_history = [[0] for _ in range(self.n_classes)]
        self.last_number_contributing_agents_history = [[1] for _ in range(self.n_classes)]
        self.upper_references_history = [[] for _ in range(self.n_classes)]
        self.lower_references_history = [[] for _ in range(self.n_classes)]

        activations, candidacies = None, None
        if activation_periods != None and triggering_events_instants != None and global_iterations_instants != None:
            activations, candidacies = activations_candidacies(self.activation_periods,
                                                            self.triggering_events_instants,
                                                            global_iterations_instants) 
        self.activations = activations
        self.candidacies = candidacies
        self.fitness_history = [np.zeros(len(global_iterations_instants[class_ind])) for class_ind in range(self.n_classes)]
        self.freshness_history = [np.zeros(len(global_iterations_instants[class_ind])) for class_ind in range(self.n_classes)]
        self.initial_fitness_tracking = [np.zeros(len(global_iterations_instants[class_ind])) for class_ind in range(self.n_classes)]
        self.local_instability_history = [[] for _ in range(self.n_classes)]

        self.n_episodes_tracker = np.zeros(self.n_classes)

        self.planet_gravity = planet_gravity

        # cheetah marionette
        self.cheetah_marionette = gym.make(environment_ids['cheetah'],
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True)
        self.cheetah_marionette.unwrapped.model.opt.gravity = [0, 0, -planet_gravity]
        cheetah_obs_dim = self.cheetah_marionette.observation_space.shape[0]
        cheetah_act_dim = self.cheetah_marionette.action_space.shape[0]
        self.cheetah_model = Network(shape_in=17, action_shape=6)
        self.cheetah_policy = Policy(self.cheetah_model)
        self.cheetah_optimizer = torch.optim.Adam(self.cheetah_model.parameters(), lr=3e-4, eps=1e-5)
        self.cheetah_runner = EnvRunner(self.cheetah_marionette, self.cheetah_policy, nsteps=2048,
                   transforms=[GAE(self.cheetah_policy, gamma=0.99, lam=0.95)])
        self.cheetah_sampler = TrajectorySampler(self.cheetah_runner, num_epochs=10, minibatch_size=64,
                            transforms=[NormalizeAdvantages()])
        self.cheetah_ppo = PPO(self.cheetah_policy, self.cheetah_optimizer)

        # ant marionette
        self.ant_marionette = gym.make(environment_ids['ant'],
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True)
        self.ant_marionette.unwrapped.model.opt.gravity = [0, 0, -planet_gravity]
        ant_obs_dim = self.ant_marionette.observation_space.shape[0]
        ant_act_dim = self.ant_marionette.action_space.shape[0]
        self.ant_model = Network(shape_in=ant_obs_dim, action_shape=ant_act_dim)
        self.ant_policy = Policy(self.ant_model)
        self.ant_optimizer = torch.optim.Adam(self.ant_model.parameters(), lr=3e-4, eps=1e-5)
        self.ant_runner = EnvRunner(self.ant_marionette, self.ant_policy, nsteps=2048,
                   transforms=[GAE(self.ant_policy, gamma=0.99, lam=0.95)])
        self.ant_sampler = TrajectorySampler(self.ant_runner, num_epochs=10, minibatch_size=64,
                            transforms=[NormalizeAdvantages()])
        self.ant_ppo = PPO(self.ant_policy, self.ant_optimizer)

        # leg marionette
        self.leg_marionette = gym.make(environment_ids['hopper'],
               ctrl_cost_weight=0.001,
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True)
        self.leg_marionette.unwrapped.model.opt.gravity = [0, 0, -planet_gravity]
        hopper_obs_dim = self.leg_marionette.observation_space.shape[0]
        hopper_act_dim = self.leg_marionette.action_space.shape[0]
        self.leg_model = Network(shape_in=hopper_obs_dim, action_shape=hopper_act_dim)
        self.leg_policy = Policy(self.leg_model)
        self.leg_optimizer = torch.optim.Adam(self.leg_model.parameters(), lr=3e-4, eps=1e-5)
        self.leg_runner = EnvRunner(self.leg_marionette, self.leg_policy, nsteps=2048,
                   transforms=[GAE(self.leg_policy, gamma=0.99, lam=0.95)])
        self.leg_sampler = TrajectorySampler(self.leg_runner, num_epochs=10, minibatch_size=64,
                            transforms=[NormalizeAdvantages()])
        self.leg_ppo = PPO(self.leg_policy, self.leg_optimizer)

        # humanoid marionette
        self.humanoid_marionette = gym.make(environment_ids['humanoid'],
               ctrl_cost_weight=0.01,
               reset_noise_scale=0.1,
               exclude_current_positions_from_observation=True)
        self.humanoid_marionette.unwrapped.model.opt.gravity = [0, 0, -planet_gravity]
        # Automatically get correct dimensions
        humanoid_obs_dim = self.humanoid_marionette.observation_space.shape[0]
        humanoid_act_dim = self.humanoid_marionette.action_space.shape[0]
        self.humanoid_model = Network(shape_in=humanoid_obs_dim, action_shape=humanoid_act_dim, hidden_size=256)
        self.humanoid_policy = Policy(self.humanoid_model)
        self.humanoid_optimizer = torch.optim.Adam(self.humanoid_model.parameters(), lr=3e-4, eps=1e-5)
        self.humanoid_runner = EnvRunner(self.humanoid_marionette, self.humanoid_policy, nsteps=2048,
                   transforms=[GAE(self.humanoid_policy, gamma=0.99, lam=0.95)])
        self.humanoid_sampler = TrajectorySampler(self.humanoid_runner, num_epochs=20, minibatch_size=32,
                            transforms=[NormalizeAdvantages()])
        self.humanoid_ppo = PPO(self.humanoid_policy, self.humanoid_optimizer)

    def is_active(self, class_ind, iteration):
        if iteration >= len(self.activations[class_ind]):
            iteration = len(self.activations[class_ind])-1
            print("activation error")
        return self.activations[class_ind][iteration] == 1


    def is_candidate(self, class_ind, iteration):
        if iteration >= len(self.candidacies[class_ind]):
            iteration = len(self.candidacies[class_ind])-1
        return self.candidacies[class_ind][iteration] == 1
    

    def training_cheetah_marionette(self, number_episodes):
        history_steps = []
        history_rewards = []
        for it in range(number_episodes*20):
            batch = self.cheetah_sampler.get_next()
            loss, gnorm = self.cheetah_ppo.step(batch)
            if it % 20 == 0:
                eval_rews = []
                for _ in range(5):
                    obs, _ = self.cheetah_marionette.reset()
                    ep_rew = 0.0
                    done = False
                    while not done:
                        act_dict = self.cheetah_policy.act(obs, training=False)
                        obs, r, term, trunc, _ = self.cheetah_marionette.step(act_dict['actions'])
                        ep_rew += r
                        done = term or trunc
                    eval_rews.append(ep_rew)
                mean_rew = np.mean(eval_rews)

                history_steps.append(it * 2048)
                history_rewards.append(mean_rew)
        return history_rewards

    def training_ant_marionette(self, number_episodes):
        history_steps = []
        history_rewards = []
        for it in range(number_episodes*20):  # Increased to give it more time
            batch = self.ant_sampler.get_next()
            loss, gnorm = self.ant_ppo.step(batch)
            if it % 20 == 0:
                eval_rews = []
                for _ in range(5):
                    obs, _ = self.ant_marionette.reset()
                    ep_rew = 0.0
                    done = False
                    while not done:
                        act_dict = self.ant_policy.act(obs, training=False)
                        obs, r, term, trunc, _ = self.ant_marionette.step(act_dict['actions'])
                        ep_rew += r
                        done = term or trunc
                    eval_rews.append(ep_rew)
                mean_rew = np.mean(eval_rews)
                history_steps.append(it * 2048)
                history_rewards.append(mean_rew)
        return history_rewards
    
    def training_leg_marionette(self, number_episodes):
        history_steps = []
        history_rewards = []
        for it in range(number_episodes*20):  # Increased to give it more time
            batch = self.leg_sampler.get_next()
            loss, gnorm = self.leg_ppo.step(batch)
            if it % 20 == 0:
                eval_rews = []
                for _ in range(5):
                    obs, _ = self.leg_marionette.reset()
                    ep_rew = 0.0
                    done = False
                    while not done:
                        act_dict = self.leg_policy.act(obs, training=False)
                        obs, r, term, trunc, _ = self.leg_marionette.step(act_dict['actions'])
                        ep_rew += r
                        done = term or trunc
                    eval_rews.append(ep_rew)
                mean_rew = np.mean(eval_rews)
                history_steps.append(it * 2048)
                history_rewards.append(mean_rew)
        return history_rewards
    
    def training_humanoid_marionette(self, number_episodes):
        history_steps = []
        history_rewards = []
        for it in range(20*number_episodes):  # Increased to give it more time
            batch = self.humanoid_sampler.get_next()
            loss, gnorm = self.humanoid_ppo.step(batch)
            if it % 20 == 0:
                eval_rews = []
                for _ in range(5):
                    obs, _ = self.humanoid_marionette.reset()
                    ep_rew = 0.0
                    done = False
                    while not done:
                        act_dict = self.humanoid_policy.act(obs, training=False)
                        obs, r, term, trunc, _ = self.humanoid_marionette.step(act_dict['actions'])
                        ep_rew += r
                        done = term or trunc
                    eval_rews.append(ep_rew)
                mean_rew = np.mean(eval_rews)
                history_steps.append(it * 2048)
                history_rewards.append(mean_rew)
        return history_rewards

    def model_based_training_outcome(self, class_id, global_policy_fitness, number_contributing_agents, number_episodes):
        features = np.array([
            global_policy_fitness,
            number_contributing_agents,
            self.planet_gravity,
            1.0,
        ])
        initialization, log_reversion, log_long_term, log_sigma = (
            features @ self.Xi_matrix[CLASS_NAMES[class_id]]
        )
        mean_reversion_rate = np.exp(log_reversion)
        long_term_cumulative_reward = np.exp(log_long_term)
        sigma = np.exp(log_sigma)

        ################################################################################################################
        # self.n_episodes_tracker[class_id] += number_episodes

        model = OrnsteinUhlenbeck()
        model.params = np.array([mean_reversion_rate, long_term_cumulative_reward, sigma])
        # simulator = Simulator1D(initialization, self.n_episodes_tracker[class_id], 1, model)
        simulator = Simulator1D(initialization, number_episodes, 1, model)
        rewards = simulator.sim_path().flatten()
        return rewards #[number_episodes+1: self.n_episodes_tracker[class_id]+1]
    
    def model_free_training_outcome(self, class_id, number_episodes):
        # interact with self.local_environment via self.model
        rewards = []
        if class_id == 0:
            rewards = self.training_cheetah_marionette(number_episodes)
        if class_id == 1:
            rewards = self.training_ant_marionette(number_episodes)
        if class_id == 2:
            rewards = self.training_leg_marionette(number_episodes)
        if class_id == 3:
            rewards = self.training_humanoid_marionette(number_episodes)
        return rewards

    def step(self, 
             class_id, 
             global_iteration_index, 
             global_iteration_instant, 
             threshold, 
             global_policy_fitness,
             simualated_rewards = True):
        if len(self.triggering_events_instants[class_id]) == 0:
            return None
        last_triggering_event_index = len([z for z in self.triggering_events_instants[class_id] if z <= global_iteration_instant]) - 1
        n_episodes = self.number_of_episodes[class_id][last_triggering_event_index]
        if n_episodes == 0:
            return None
        freshness = self.last_global_iteration_indices[class_id] - global_iteration_index + 1
        self.last_global_iteration_indices[class_id] = global_iteration_index

        rewards = []
        if simualated_rewards:
            rewards = self.model_based_training_outcome(class_id,
                                                    self.last_global_fitness_values_history[class_id][-1], 
                                                    self.last_number_contributing_agents_history[class_id][-1], 
                                                    n_episodes)
        else:
            rewards = self.model_free_training_outcome(class_id, self.number_of_episodes[class_id][last_triggering_event_index])
        
        self.rewards_history[class_id].append(rewards)
        self.lower_references_history[class_id].append(np.min(rewards))
        local_fitness = np.mean(rewards) * np.exp(self.temperature*freshness)

        self.fitness_history[class_id][global_iteration_index] = local_fitness
        self.freshness_history[class_id][global_iteration_index] = freshness

        upload_condition = local_fitness - global_policy_fitness > threshold or global_iteration_index == 0 
        # direct upload in the first iteration
        
        if upload_condition:
            model = None
            if class_id == 0:
                model = self.cheetah_model
            if class_id == 1:
                model = self.ant_model
            if class_id == 2:
                model = self.leg_model
            if class_id == 3:
                model = self.humanoid_model
            # Only the upload is quantized; the local training model stays float32.
            payload = quantize_model(model)
            return (freshness, local_fitness, payload, n_episodes, max(rewards))
        return max(rewards)
    
    def update(self,
               class_id,
               new_global_policy_fitness,
               new_number_contributing_agents,
               upper_reference,
               update_model = False):
        # performed if the aggregator sends back the model
        self.upper_references_history[class_id].append(upper_reference)

        if update_model == True:
            self.last_global_fitness_values_history[class_id].append(new_global_policy_fitness)
            self.last_number_contributing_agents_history[class_id].append(new_number_contributing_agents)
            self.n_episodes_tracker[class_id] = 0

        # the most difficult part: the instability increment
        #lower_reference = self.last_global_fitness_values_history[class_id][-1] #alternative1
        if len(self.lower_references_history[class_id]) < 2: 
            return 0
            
        lower_reference = min(self.lower_references_history[class_id][-2], self.lower_references_history[class_id][-1])
        local_instability = 0

        if len(self.rewards_history[class_id]) != 0 and upper_reference > lower_reference:
            scaled_rewards = (self.rewards_history[class_id][-1] - lower_reference)/(upper_reference - lower_reference)
            local_instability = np.mean(np.exp(-self.scaling_constant*scaled_rewards))

        self.local_instability_history[class_id].append(local_instability)
        return local_instability
    
    def update_model(self, class_id, model):
        if class_id == 0:
            self.cheetah_model = copy.deepcopy(model)
            #print("cheetah model updated")
            self.cheetah_policy = Policy(self.cheetah_model)
            self.cheetah_optimizer = torch.optim.Adam(self.cheetah_model.parameters(), lr=3e-4, eps=1e-5)
            self.cheetah_runner = EnvRunner(self.cheetah_marionette, self.cheetah_policy, nsteps=2048,
                    transforms=[GAE(self.cheetah_policy, gamma=0.99, lam=0.95)])
            self.cheetah_sampler = TrajectorySampler(self.cheetah_runner, num_epochs=10, minibatch_size=64,
                                transforms=[NormalizeAdvantages()])
            self.cheetah_ppo = PPO(self.cheetah_policy, self.cheetah_optimizer)
            
        elif class_id == 1:
            self.ant_model = copy.deepcopy(model)
            self.ant_policy = Policy(self.ant_model)
            self.ant_optimizer = torch.optim.Adam(self.ant_model.parameters(), lr=3e-4, eps=1e-5)
            self.ant_runner = EnvRunner(self.ant_marionette, self.ant_policy, nsteps=2048,
                    transforms=[GAE(self.ant_policy, gamma=0.99, lam=0.95)])
            self.ant_sampler = TrajectorySampler(self.ant_runner, num_epochs=10, minibatch_size=64,
                                transforms=[NormalizeAdvantages()])
            self.ant_ppo = PPO(self.ant_policy, self.ant_optimizer)
            #print("ant model updated")

        elif class_id == 2:
            self.leg_model = copy.deepcopy(model)
            self.leg_policy = Policy(self.leg_model)
            self.leg_optimizer = torch.optim.Adam(self.leg_model.parameters(), lr=3e-4, eps=1e-5)
            self.leg_runner = EnvRunner(self.leg_marionette, self.leg_policy, nsteps=2048,
                    transforms=[GAE(self.leg_policy, gamma=0.99, lam=0.95)])
            self.leg_sampler = TrajectorySampler(self.leg_runner, num_epochs=10, minibatch_size=64,
                                transforms=[NormalizeAdvantages()])
            self.leg_ppo = PPO(self.leg_policy, self.leg_optimizer)
            #print("leg model updated")
        else:
            self.humanoid_model = copy.deepcopy(model)
            self.humanoid_policy = Policy(self.humanoid_model)
            self.humanoid_optimizer = torch.optim.Adam(self.humanoid_model.parameters(), lr=3e-4, eps=1e-5)
            self.humanoid_runner = EnvRunner(self.humanoid_marionette, self.humanoid_policy, nsteps=2048,
                    transforms=[GAE(self.humanoid_policy, gamma=0.99, lam=0.95)])
            self.humanoid_sampler = TrajectorySampler(self.humanoid_runner, num_epochs=20, minibatch_size=32,
                                transforms=[NormalizeAdvantages()])
            self.humanoid_ppo = PPO(self.humanoid_policy, self.humanoid_optimizer)
            #print("humanoid model updated")
