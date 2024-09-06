import numpy as np
from bayes_opt import BayesianOptimization
from numpy.random import RandomState

from typing import Callable, Dict, Tuple, Optional, Union, List

class BayesianOptimizer:
    def __init__(
        self,
        function: Callable,
        pbounds: Dict[str, Tuple[int]],
        random_state: Optional[Union[int, RandomState, None]]=None,
        verbose: bool=False
        ) -> None:
        """
        BayesianOptimizer is a wrapper around the BayesianOptimization class.

        :param function: Function to optimize
        :param pbounds: Bounding boxes of the parameters, e.g., {'x': (2, 4), 'y': (-3, 3)}
        :param random_state: Seed for the parameters
        :param verbose: Verbose level at maximazation
        """
        self.optimizer = BayesianOptimization(
            f=function,
            pbounds=pbounds,
            verbose=verbose,
            random_state=random_state
        )

    def maximize(self, n_iter:int, init_points: int) -> Dict[str, float]:
        """
        Minimizing the function by changing the values of the parameters

        :param n_iter: steps of bayesian optimization
        :param init_points: Steps of random exploration

        :return: Dictionary containing the function values for the
        parameters of each iteration
        """
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

        return self.optimizer.res
    
    def get_max(self) -> Dict[str, float]:
        """
        Return the parameters that optimized the function

        :return: The optimal parameters
        """
        return self.optimizer.max['params']
    
class GeneticOptimizer:
    def __init__(
        self,
        fit_function: Callable,
        pbounds: Dict[str, Tuple[int]],
        pop_size: int,
        random_state: Optional[Union[int, RandomState, None]]=None,
        crossover_rate: float=0.8,
        mutation_rate: float=0.1,
        mutation_scale: float=0.1,
        elite_size: int=5,
        verbose: bool=False
        ) -> None:
        """
        :param fit_function: Fitness function to optimize
        :param pbounds: Bounding boxes of the parameters, e.g., {'x': (2, 4), 'y': (-3, 3)}
        :param pop_size: Number of individuals in the population
        :param random_state: Seed for the parameters
        :param verbose: Verbose level at maximazation
        """
        assert elite_size < pop_size, "Elite size must be less than the population size"
        assert elite_size > 0, "Elite size must be greater than 0"
        assert elite_size % 2 == 0, "Elite size must be an even number"
        
        self.fit_function = fit_function
        self.pbounds = pbounds
        self.pop_size = pop_size
        self.random_state = random_state
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_size = elite_size
        self.verbose = verbose
    
    def initialize_population(self) -> List[Dict[str, float]]:
        """
        Initialize the population with random values
        """
        return [{
            key: np.random.uniform(low, high)
            for key, (low, high) in self.pbounds.items()
        } for _ in range(self.pop_size)]
    
    # Selection
    def select_parents(self, population, fitness):
        probabilities = fitness / np.sum(fitness)
        return np.random.choice(population, size=2, p=probabilities)

    # Crossover
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand()
            for key in parent1.keys():
                parent1[key] = self.crossover_gene(parent1[key], parent2[key], alpha)
                
        return parent1

    def crossover_gene(self, gene1, gene2, alpha):
        return alpha * gene1 + (1 - alpha) * gene2

    # Mutation
    def mutate(self, offspring):
        if np.random.rand() < self.mutation_rate:
            # print('mutation')
            # print(offspring)
            # mutation = np.random.uniform(-1, 1)
            for key in offspring.keys():
                # Apply a small Gaussian mutation
                mutation = np.random.normal(-self.mutation_scale, self.mutation_scale)
                offspring[key] += mutation
                
                # mutation = np.random.uniform(self.pbounds[key][0], self.pbounds[key][1]) * np.random.uniform(-1, 1)
                # offspring[key] += mutation
                
                offspring[key] = np.clip(offspring[key], self.pbounds[key][0], self.pbounds[key][1])
            
            # print(offspring)
            
        return offspring

    def optimize(self, epochs=100) -> Dict[str, float]:
        """
        Minimizing the function by changing the values of the parameters

        :return: Dictionary containing the function values for the
        parameters of each iteration
        """
        population = self.initialize_population()
        
        best_params = None
        best_fitness_score = np.inf
        
        for i in range(epochs):
            fitness = np.array([self.fit_function(**pop) for pop in population])
            
            # sort population by fitness
            sorted_indices = np.argsort(fitness)
            best_individuals = [population[i] for i in sorted_indices[:self.elite_size]]
            best_fitness = fitness[sorted_indices[:self.elite_size]]
            
            new_population = list(best_individuals)
    
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents(best_individuals, best_fitness)
                _offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(_offspring)
                
                new_population.append(offspring)
            
            population = np.array(new_population)
            
            # print(f'population: {len(population)}')
            # print(f'fitness: {len(fitness)}')
            # print(f'average fitness: {np.mean(fitness)}')
            print(population)
        
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_fitness_score:
                best_fitness_score = fitness[best_index]
                best_params = population[best_index]
        
            if self.verbose:
                print(f'Epoch {i}:\tBest Score: {fitness[best_index]}\t|\tAverage Score: {np.mean(fitness)}\t|\tBest Params: {population[best_index]}')
        
        # best_index = np.argmin(fitness)
        # best_params = population[best_index]
        
        return best_params, best_fitness_score

# if __name__ == '__main__':
#     def loss_function(x, b):
#         return x**2 + b**2 + 1
    
#     optimizer = GeneticOptimizer(
#         fit_function=loss_function,
#         pbounds={'x': (0, 4), 'b': (0, 4)},
#         pop_size=10,
#         mutation_rate=0.4,
#         mutation_scale=0.5,
#         verbose=True,
#     )
    
#     print(optimizer.optimize(epochs=5))