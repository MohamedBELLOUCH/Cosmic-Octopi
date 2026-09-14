import numpy as np
import utils
import importlib
importlib.reload(utils)
from utils import federated_average
from utils import get_model_size_in_kb
import threading


from utils import Network

class Oracle:
    def __init__(self, octopi, N_classes, global_iterations_instants, hyper_params_):
        self.octopi = octopi
        self.global_iterations_instants = global_iterations_instants
        self.active_octopi_history = [[] for _ in range(N_classes)]
        self.candidate_octopi_history = [[] for _ in range(N_classes)]
        self.contributing_octopi_history = [[] for _ in range(N_classes)]
        self.global_fitness_history = [[-np.inf] for _ in range(N_classes)]
        self.last_n_contributing_octopi = np.zeros(N_classes)
        self.temperature = hyper_params_['temperature']
        self.global_upper_reference_history = [[] for _ in range(N_classes)]

        self.overhead_increments_history = [[] for _ in range(N_classes)]
        self.instability_increments_history = [[] for _ in range(N_classes)]

    def step(self, class_id, k, threshold, simulated_rewards = True, n_parallel = 1):

        Active_octopi = []
        Candidate_octopi = []
        Contributing_octopi = []
        Lagged_octopi = []
        candidate_octopi_repsonses = []
        contributing_octopi_responses = []

        for octopus in self.octopi:
            if octopus.is_active(class_id, k): 
                Active_octopi.append(octopus)
                
            if octopus.is_candidate(class_id, k):
                Candidate_octopi.append(octopus)

        def oracle_request(octopi_batch, 
                           class_id, 
                           global_iteration_index, 
                           global_iteration_instant, 
                           fitness_margin, 
                           fitness,
                           simulated_rewards_,
                           candidate_responses_,
                           contributing_octopi_,
                           contributing_responses_,
                           lagged_octopi_
                           ):

            for octopus in octopi_batch:
                response = octopus.step(class_id,
                               global_iteration_index,
                               global_iteration_instant,
                               fitness_margin, 
                               fitness,
                               simulated_rewards_)
                
                if response != None:
                    candidate_responses_.append(response)

                    if type(response) == tuple:
                        # the octopus is a contributing agent
                        contributing_octopi_.append(octopus)
                        contributing_responses_.append(response)
                        if response[1] < -1:
                            lagged_octopi_.append(octopus)
        octopi_batches = [Candidate_octopi[i: i+n_parallel] for i in range(0, len(Candidate_octopi), n_parallel)]
        
        threads = []
        for batch in octopi_batches:
            t = threading.Thread(
                target=oracle_request, 
                args=(batch,
                    class_id, 
                    k, 
                    self.global_iterations_instants[class_id][k],
                    threshold, 
                    self.global_fitness_history[class_id][-1],
                    simulated_rewards,
                    candidate_octopi_repsonses,
                    Contributing_octopi,
                    contributing_octopi_responses,
                    Lagged_octopi
                    ))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if len(candidate_octopi_repsonses) == 0:
            # end the round if no response is received from the octopi
            return None
        
        #Recap.: the response of a contributing octopus is of the form : (freshness, fitness, model, n_episodes, max_rewards)
        contributing_octopi_upper_max_reward = [r[-1] for r in contributing_octopi_responses]
        non_contributing_octopi_upper_max_reward = [r for r in candidate_octopi_repsonses if type(r) != tuple]
        contributing_octopi_models = [r[2] for r in contributing_octopi_responses]
        contributing_octopi_n_episodes = [r[3] for r in contributing_octopi_responses]

        upper_reference = max(contributing_octopi_upper_max_reward + non_contributing_octopi_upper_max_reward)
        self.global_upper_reference_history[class_id].append(upper_reference)

        # Compute global fitness
        F = self.global_fitness_history[class_id][-1]
        local_instability_increments = []
        local_instability_increment = np.nan
        if len(contributing_octopi_responses) > 0:
            global_model = federated_average(contributing_octopi_models, contributing_octopi_n_episodes)
            fitnesses = [response[1] for response in contributing_octopi_responses]
            F = np.sum(fitnesses)/np.sum([np.exp(self.temperature*response[0]) for response in contributing_octopi_responses])
            #broadcast to the candidate octopi
            for octopus in Candidate_octopi:
                octopus.update_model(class_id, global_model)
                local_instability_increment = octopus.update(class_id, F, len(contributing_octopi_responses), upper_reference, True)
                if local_instability_increment != np.nan:
                    local_instability_increments.append(local_instability_increment)
        else:
            for octopus in Candidate_octopi:
                if octopus in Lagged_octopi:
                    local_instability_increment = octopus.update(class_id, F, 
                                                       len(self.contributing_octopi_history[class_id][-1]), 
                                                       upper_reference, True)
                    octopus.update_model(class_id, global_model)
                else:
                    local_instability_increment = octopus.update(class_id, F, 
                                                        len(self.contributing_octopi_history[class_id][-1]), 
                                                        upper_reference, False)
                if local_instability_increment != np.nan:
                    local_instability_increments.append(local_instability_increment)
                
        overhead_increment = np.sum([get_model_size_in_kb(model) for model in contributing_octopi_models])
        instability_increment = np.mean(local_instability_increments)

        self.global_fitness_history[class_id].append(F) # duplicated in case the contributing_octopi_responses is empty
        self.active_octopi_history[class_id].append(Active_octopi) 
        # just to avoid duplication, octopi activation is not class dependent
        self.candidate_octopi_history[class_id].append(Candidate_octopi)
        self.contributing_octopi_history[class_id].append(Contributing_octopi)

        self.overhead_increments_history[class_id].append(overhead_increment)
        self.instability_increments_history[class_id].append(instability_increment)

        return overhead_increment, instability_increment