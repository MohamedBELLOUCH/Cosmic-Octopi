import utils
import octopus
import oracle
import importlib
importlib.reload(utils)
importlib.reload(octopus)
importlib.reload(oracle)

from utils import generate_global_iterations_instants_reductionist
from utils import generate_global_iterations_instants_holistic
from utils import generate_learning_periods
from utils import generate_triggering_events_instants
from octopus import Octopus
from oracle import Oracle
import numpy as np

class_labels = ['Cheetah', 'Ant', 'Leg', 'Humanoid']



def blackbox_reductionist(class_ind,
                          fitness_margins_sequence,
                          system_dynamics_params_, 
                          diffusion_params_,
                          hyper_params_, 
                          n_parallel = 1,
                          verbose = True
                        ):
    
    horizon = len(fitness_margins_sequence)

    global_iterations_instants, oracle_iterations, time_horizon = generate_global_iterations_instants_reductionist(
        system_dynamics_params_, 
        horizon
        )

    activation_periods, _ = generate_learning_periods(
        system_dynamics_params_, 
        time_horizon
        )

    triggering_events_instants, number_of_episodes = generate_triggering_events_instants(
        system_dynamics_params_,
        global_iterations_instants,
        activation_periods
        )
    
    oracle_iterations_scenario = [iteration for iteration in oracle_iterations if iteration[0] == class_ind][:horizon]

    g = 9.81
    Octopi = [
        Octopus(
            i,
            activation_periods[i],
            triggering_events_instants[i],
            number_of_episodes[i],
            global_iterations_instants,
            diffusion_params_[i],
            hyper_params_,
            np.random.gamma(g**2 / system_dynamics_params_['gamma_grav'], system_dynamics_params_['gamma_grav'] / g)
        )
        for i in range(system_dynamics_params_['n_octopi'])
    ]

    galactic_oracle = Oracle(
        Octopi,
        system_dynamics_params_['n_classes'],
        global_iterations_instants,
        hyper_params_,
    )

    overhead_increments_tracker = []
    instability_increments_tracker = []

    if verbose: print("Initialiation terminated")

    for iteration in oracle_iterations_scenario:
        fitness_margin = fitness_margins_sequence[iteration[1]]
        if verbose:
            print(
                f"Reductionist blackbox | {class_labels[class_ind]} | "
                f"Iteration {iteration[1]}"
                )
        res = galactic_oracle.step(
            iteration[0],
            iteration[1],
            fitness_margin,
            simulated_rewards=True,
            n_parallel=n_parallel
        )
        if res != None:
            overhead_increments_tracker.append(res[0])
            instability_increments_tracker.append(res[1])

    overhead = sum(overhead_increments_tracker)
    instability = sum(instability_increments_tracker)

    return overhead, instability


def blackbox_holistic(sequence,
                    system_dynamics_params_, 
                    diffusion_params_,
                    hyper_params_,
                    fitness_margin_bounds_,
                    n_parallel = 1,
                    verbose = True
                    ):
    
    horizon = len(sequence)

    global_iterations_instants, oracle_iterations, time_horizon = generate_global_iterations_instants_holistic(
        system_dynamics_params_, 
        horizon
        )

    activation_periods, _ = generate_learning_periods(
        system_dynamics_params_, 
        time_horizon
        )

    triggering_events_instants, number_of_episodes = generate_triggering_events_instants(
        system_dynamics_params_,
        global_iterations_instants,
        activation_periods
        )
    
    # oracle_iterations_scenario = [iteration for iteration in oracle_iterations if iteration[0] == class_ind][:horizon]

    g = 9.81
    Octopi = [
        Octopus(
            i,
            activation_periods[i],
            triggering_events_instants[i],
            number_of_episodes[i],
            global_iterations_instants,
            diffusion_params_[i],
            hyper_params_,
            np.random.gamma(g**2 / system_dynamics_params_['gamma_grav'], system_dynamics_params_['gamma_grav'] / g)
        )
        for i in range(system_dynamics_params_['n_octopi'])
    ]

    galactic_oracle = Oracle(
        Octopi,
        system_dynamics_params_['n_classes'],
        global_iterations_instants,
        hyper_params_,
    )

    overhead_increments_tracker = []
    instability_increments_tracker = []

    if verbose: print("Initialiation terminated")

    for iteration in oracle_iterations:
        min_fitness_margin = fitness_margin_bounds_[iteration[0]][0]
        max_fitness_margin = fitness_margin_bounds_[iteration[0]][1]
        fitness_margin = sequence[iteration[0]]*(max_fitness_margin - min_fitness_margin) + min_fitness_margin
        if verbose:
            print(
                f"Holistic blackbox | {class_labels[iteration[0]]} "
                f"Iteration {iteration[1]}"
                )
        res = galactic_oracle.step(
            iteration[0],
            iteration[1],
            fitness_margin,
            simulated_rewards=True,
            n_parallel=n_parallel
        )
        if res != None:
            overhead_increments_tracker.append(res[0])
            instability_increments_tracker.append(res[1])

    overhead = sum(overhead_increments_tracker)
    instability = sum(instability_increments_tracker)

    return overhead, instability