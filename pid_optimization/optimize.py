import os
import pandas as pd
import numpy as np
from collections import namedtuple

from .optimizer import BayesianOptimizer, GeneticOptimizer
from controllers.pid import Controller as PIDController
from tinyphysics import run_rollout_optim

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])

def optimize_pid_controller_bayesian():
    """
    Optimize the PID controller using Bayesian Optimization
    """
    
    # List to hold individual dataframes
    dataframes = []

    # Read all CSV files in the directory
    directory_path = os.path.join("processed_data","data")
    for filename in os.listdir(directory_path)[:5]:
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes into a single dataframe
    data = pd.concat(dataframes, ignore_index=True)
    
    def loss_function_1(p, i, d):
        pid = PIDController(p, i, d)
        
        data['pid_action'] = data.apply(
                lambda x: pid.update(
                    x['target'],
                    x['current'],
                    State(
                        roll_lataccel=x['roll'],
                        v_ego=x['vEgo'],
                        a_ego=x['aEgo'],
                    ),
                    [],
                ),
                axis=1,
            )
        
        # calculate rmse between pid_action and cmd
        rmse = np.sqrt(np.mean((data['pid_action'] - data['cmd'])**2))
        
        return -rmse

    def loss_function_2 (p, i, d):
        # run the simulation for every run and compute total cost (lat_accel_cost * multiplier) + jerk_cost
        
        pid = PIDController(p, i, d)
        total_cost = 0
        
        directory_path = "data"
        files = os.listdir(directory_path)
        files.sort()
        
        batch_size = 4
        # pick random files from the directory
        files = list(np.random.choice(files, batch_size))
        # print(files)
        
        for index, file in enumerate(files):            
            cost, _, _ = run_rollout_optim(
                data_path=os.path.join(directory_path, file),
                controller=pid,
                model_path="./models/tinyphysics.onnx",
                debug=False,
            )
            
            total_cost += cost['total_cost']
            
        avg_cost = total_cost / batch_size
        
        return -avg_cost

    optimizer = BayesianOptimizer(
        function=loss_function_2,
        pbounds={
            'p': (-0.1, 0.1),
            'i': (-0.2, 0.2),
            'd': (-0.1, 0.1),
        },
        verbose=2,
        random_state=None,
    )
    
    optimizer.maximize(n_iter=150, init_points=30)
    
    optimal_params = optimizer.get_max()
    
    return optimal_params

def optimize_pid_controller_genetic():
    """
    Optimize the PID controller using Genetic Algorithm
    """
    
    def loss_function(p, i, d):
        
        pid = PIDController(p, i, d)
        total_cost = 0
        
        directory_path = "data"
        files = os.listdir(directory_path)
        files.sort()
        
        batch_size = 4
        # pick random files from the directory
        files = list(np.random.choice(files, batch_size))
        # print(files)
        
        for index, file in enumerate(files):            
            cost, _, _ = run_rollout_optim(
                data_path=os.path.join(directory_path, file),
                controller=pid,
                model_path="./models/tinyphysics.onnx",
                debug=False,
            )
            
            total_cost += cost['total_cost']
            
        avg_cost = total_cost / batch_size
        
        return avg_cost

    optimizer = GeneticOptimizer(
        fit_function=loss_function,
        pbounds={
            'p': (-0.25, 0.25),
            'i': (-0.2, 0.2),
            'd': (-0.1, 0.1),
        },
        pop_size=4,
        crossover_rate=1,
        mutation_rate=1,
        mutation_scale=0.1,
        elite_size=2,
        random_state=None,
        verbose=True,
    )
    
    optimal_params = optimizer.optimize(epochs=10)
    
    return optimal_params