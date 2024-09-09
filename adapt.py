"""
Map-Based Bayesian Optimisation (MBOA) for Hexapod Robot Control

This script implements Map-Based Bayesian Optimisation experiments for hexapod robot control.
It uses a pre-generated map of behaviors and their expected performances to guide the
optimization process, aiming to find the best controller for the robot.

The code is adapted from https://github.com/chrismailer/mailer_gecco_2021

Main components:
1. Data loading functions for centroids, map, and genomes
2. Upper Confidence Bound (UCB) acquisition function
3. MBOA algorithm implementation
4. Utility functions for Gaussian Process regression

Dependencies:
- numpy
- GPy
- hexapod (custom module for robot simulation)

Usage:
Run this script as the main program to execute the MBOA algorithm.
Modify the parameters in the __main__ section as needed.
"""

from copy import copy
import numpy as np
import GPy
from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
import neat
import pickle

def evaluate_gait(genome, scenario: list, duration=5 ):
    """
    Evaluate a single genome using the hexapod simulator.
    
    Args:
        genome: The genome to evaluate.
        duration (float): Duration of the simulation in seconds.
    
    Returns:
        float: The fitness of the genome (distance traveled).
    """
    # Load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'NEATHex/config-feedforward')
    
    # Create neural network from genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Set up controller
    leg_params = np.array(stationary).reshape(6, 5)
    controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi/6, ann=net)
    
    # Set up and run simulator
    simulator = Simulator(controller=controller, visualiser=True, collision_fatal=True, failed_legs=scenario)
    
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            return 0  # Return 0 fitness if collision occurs
    
    # Calculate fitness (distance traveled along x-axis)
    fitness = simulator.base_pos()[0]
    
    simulator.terminate()
    return fitness
import neat 
import pickle

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

def load_centroids(filename: str) -> np.ndarray:
    """
    Load CVT voronoi centroids from a file.

    Args:
        filename (str): Path to the file containing centroids.

    Returns:
        np.ndarray: Array of centroids.
    """
    return np.loadtxt(filename)

def load_map(filename: str, genomes_filename: str, dim: int = 6) -> tuple:
    """
    Load the generated map, including fitness, descriptors, and genomes.

    Args:
        filename (str): Path to the map file.
        genomes_filename (str): Path to the genomes file (.pkl).
        dim (int): Dimension of the descriptor space.

    Returns:
        tuple: (fitness array, descriptor array, genomes array)
    """
    data = np.loadtxt(filename)
    fit = data[:, 0]
    desc = data[:, 1:dim + 1]
    
    # Load genomes from .pkl file
    with open(genomes_filename, 'rb') as f:
        genomes = pickle.load(f)
    
    return fit, desc, genomes

def UCB(mu_map: np.ndarray, kappa: float, sigma_map: np.ndarray) -> int:
    """
    Upper Confidence Bound acquisition function for Bayesian optimization.

    Args:
        mu_map (np.ndarray): Mean predictions.
        kappa (float): Exploration-exploitation trade-off parameter.
        sigma_map (np.ndarray): Standard deviation predictions.

    Returns:
        int: Index of the point with the highest UCB value.
    """
    GP = mu_map + kappa * sigma_map
    return np.argmax(GP)

def MBOA(map_filename: str, genomes_filename: str, centroids_filename: str, eval_func, 
         scenario: list,
         max_iter: int, rho: float = 0.4, print_output: bool = True) -> tuple:
    """
    Map-Based Bayesian Optimisation algorithm.

    Args:
        map_filename (str): Path to the map file.
        genomes_filename (str): Path to the genomes file.
        centroids_filename (str): Path to the centroids file.
        eval_func (callable): Function to evaluate real performance.
        max_iter (int): Maximum number of iterations.
        rho (float): Length scale for the GP kernel.
        print_output (bool): Whether to print progress information.

    Returns:
        tuple: (number of iterations, best index, best performance, updated map)
    """
    alpha = 0.90
    kappa = 0.05
    variance_noise_square = 0.001
    dim_x = 6

    num_it = 0
    real_perfs, tested_indexes = [-1], []
    X, Y = np.empty((0, dim_x)), np.empty((0, 1))

    centroids = load_centroids(centroids_filename)
    fits, descs, ctrls = load_map(map_filename, genomes_filename, centroids.shape[1])

    n_fits, n_descs, n_ctrls = np.array(fits), np.array(descs), np.array(ctrls)
    n_fits_real = copy(n_fits)
    fits_saved = copy(n_fits)

    started = False

    while (max(real_perfs) < alpha * max(n_fits_real)) and (num_it <= max_iter):
        if started:
            kernel = GPy.kern.Matern52(dim_x, lengthscale=rho, ARD=False) + GPy.kern.White(dim_x, np.sqrt(variance_noise_square))
            m = GPy.models.GPRegression(X, Y, kernel)
            means, variances = m.predict(n_descs)
            n_fits_real = means.flatten() + fits_saved
            index_to_test = UCB(n_fits_real, kappa, variances.flatten())
        else:
            index_to_test = np.argmax(n_fits)
            started = True
            real_perfs = []

        if print_output:
            print(f"Expected perf: {n_fits_real[index_to_test]}")

        if index_to_test in tested_indexes:
            if print_output:
                print("Behaviour already tested")
            break

        ctrl_to_test = n_ctrls[index_to_test]
        tested_indexes.append(index_to_test)

        real_perf = eval_func(ctrl_to_test, scenario)
        if print_output:
            print(f"Real perf: {real_perf}")

        num_it += 1

        X = np.vstack((X, n_descs[index_to_test]))
        Y = np.vstack((Y, np.array(real_perf) - fits_saved[index_to_test]))

        real_perfs.append(real_perf)

        new_map = np.loadtxt(map_filename)
        new_map[:, 0] = n_fits_real

    best_index = tested_indexes[np.argmax(real_perfs)]
    best_perf = max(real_perfs)

    return num_it, best_index, best_perf, new_map

 # NEAT evaluation function
def evaluate_gait(x, failed_legs = [], duration=5):
    net = neat.nn.FeedForwardNetwork.create(x, config)
    # Reset net

    leg_params = np.array(stationary).reshape(6, 5)
    # Set up controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                ann=net)
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=True, collision_fatal=False,
                            failed_legs=failed_legs)
    # Step in simulator
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
                                posinf=0.0, neginf=0.0)
    # Terminate Simulator
    simulator.terminate()
    # Assign fitness to genome
    x.fitness = fitness
    return fitness

if __name__ == "__main__":
    print("Running Map-Based Bayesian Optimisation with Hexapod Simulator")
    
    map_file = "mapElitesOutput/NEAT/1_10000archive/archive1001510.dat"
    genomes_file = "mapElitesOutput/NEAT/1_10000archive/archive_genome1001510.pkl"
    centroids_file = "centroids/centroids_10000_6.dat"
    scenarios = [
        [[]],
        [[1], [2], [3], [4], [5], [6]],
        [[1, 4], [2, 5], [3, 6]],
        [[1, 3], [2, 4], [3, 5], [4, 6], [5, 1], [6, 2]],
        [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario}")
        for i in range(len(scenario)):
            print(f"Subscenario {i+1}: {scenario[i]}")
             # Use the evaluate_gait function as the evaluation function for MBOA
            iterations, best_idx, best_performance, updated_map = MBOA(
                map_file, genomes_file, centroids_file, evaluate_gait, 
                scenario[i],
                max_iter=100, rho=0.4, print_output=True
            )
    

    print(f"Optimization completed in {iterations} iterations")
    print(f"Best index: {best_idx}, Best performance: {best_performance}")
    
    # Optionally save the updated map
    # np.savetxt("mapElitesOutput/NEAT/1_10000archive/updated_map.dat", updated_map)