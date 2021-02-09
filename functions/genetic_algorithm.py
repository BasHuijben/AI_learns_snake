import numpy as np
from operator import attrgetter

import multiprocessing
from joblib import Parallel, delayed

from functions.snake import Snake

# determine number of computer cores
num_cores = multiprocessing.cpu_count()


class GeneticAlgorithm:
    """
    Genetics algorithm for playing Snake. The genetic algorithm exists out of the following steps:
        1) Generate population, which is called a generation
        2) Play game
        3) Selected fittest snakes that will survive and will be part of the next generation
        4) Determine parents
        5) Generate children. These children will be based on the DNA of their parents (crossover) and random mutations
        6) Generate new population. New population will be based on the newly generated children and survivors
        7) repeat steps 2 till 6

    Attributes:
        population_size (int): Population size
        survival_perc (float): Survival percentages. Value between 0 and 1
        mutation_rate (float): Mutation percentages. Value between 0 and 1
        parent_perc (float): Parent percentages. Value between 0 and 1
        population (list): List of snake objects. Contains all the snakes in the population
    """

    def __init__(self, population_size, survival_perc, parent_perc, mutation_rate):
        """ Initialize object

        Args:
            population_size (int): Population size
            survival_perc (float): Survival percentages. Value between 0 and 1
            parent_perc (float): Parent percentages. Value between 0 and 1
            mutation_rate (float): Mutation percentages. Value between 0 and 1
        """
        self.population_size = population_size
        self.survival_perc = survival_perc
        self.mutation_rate = mutation_rate
        self.parent_perc = parent_perc

        self.population = None
        self.generate_population()

    def generate_population(self):
        """ Generate population

        When no population exists: create new population with random snakes
        When a population exists: create new population based on fittest snakes and children.
        """
        # initialize population
        if self.population is None:
            self.population = [Snake() for _ in range(self.population_size)]
        else:
            # Determine fittest snakes
            survivors = self.survival_of_the_fittest()

            # Determine which snakes may be parents
            parents = self.select_parents()

            # Generate children based on crossover and mutations
            children = self.generate_children(parents)

            # reset parameters of snakes. So that they can play the game again
            for survivor in survivors:
                survivor.reset_snake()

            # Create new population. Survivors + children
            self.population = survivors + children

    def play_games(self, parallel=True):
        """ Play Snake with each snake in population

        Args:
            parallel (bool): When True the games are played in parallel. Otherwise the games are played in serie
        """
        if parallel:
            self.population = Parallel(n_jobs=num_cores)(delayed(self.play_game)(snake) for snake in self.population)
        else:
            for snake in self.population:
                self.play_game(snake)

    @staticmethod
    def play_game(snake):
        """ Play Snake

        Args:
            snake (Snake): Snake object. Generated with class Snake()

        Returns:
            Snake: Snake object which has played a game of Snake
        """
        while snake.alive:
            snake.make_decision()
            snake.update_snake()
            snake.found_apple()
            snake.snake_alive()
        return snake

    def get_population_fitness(self):
        """ Get mean fitness of entire population

        Returns:
            float: Mean population fitness
        """
        return np.mean([snake.fitness for snake in self.population])

    def get_population_score(self):
        """ Get mean population score (number of apples eaten)

        Returns:
            float: Mean population score
        """
        return np.mean([snake.total_apples_found for snake in self.population])

    def get_best_fitness(self):
        """ Get fitness of best snake in population

        Returns:
            float: Fitness of best snake
        """
        return np.max([snake.fitness for snake in self.population])

    def get_best_score(self):
        """ Get score of best snake in population

        Returns:
            float: Score of best snake

        """
        return np.max([snake.total_apples_found for snake in self.population])

    def get_best_snake(self):
        """ Get best snake in population

        Returns:
            Snake: Snake with the best fitness in population
        """
        return max(self.population, key=attrgetter('fitness'))

    def survival_of_the_fittest(self):
        """ Select fittest snakes in population

        Returns:
            list: List with Snake object
        """
        n_to_survive = int(self.population_size * self.survival_perc)
        sorted_population = sorted(self.population, key=lambda snake: snake.fitness, reverse=True)
        survivors = sorted_population[:n_to_survive]
        return survivors

    def select_parents(self):
        """ Select parents. The fittest snakes will have a change to be parents

        Returns:
            list: List with Snake objects
        """
        n_parents = int(self.population_size * self.parent_perc)
        sorted_population = sorted(self.population, key=lambda snake: snake.fitness, reverse=True)
        parents = sorted_population[:n_parents]
        return parents

    def generate_children(self, survivors):
        """ Generate children based on a crossover of DNA between parents and mutations that could occur

        Args:
            survivors (list): List of Snake objects

        Returns:
            list: List of Snake object
        """
        n_new_children = self.population_size - len(survivors)
        soulmates = np.random.choice(len(survivors), size=(n_new_children, 2))
        soulmates = [(survivors[parent1], survivors[parent1]) for parent1, parent2 in soulmates]
        children = [self.crossover(parent1, parent2) for parent1, parent2 in soulmates]
        children = [self.mutation(child) for child in children]
        return children

    @staticmethod
    def crossover(parent1, parent2):
        """ Single point binary crossover between parent 1 and parent 2.

        Args:
            parent1 (Snake): Parent 1
            parent2 (Snake): Parent 2

        Returns:
            Snake: Snake with DNA based on parent 1 and 2
        """
        dna_parent1 = parent1.dna.reshape(-1)
        dna_parent2 = parent2.dna.reshape(-1)

        random_position = np.random.choice(dna_parent1.shape[0])
        dna_child = np.concatenate([dna_parent1[:random_position], dna_parent2[random_position:]])

        snake_child = Snake()
        snake_child.dna = np.asarray(dna_child).reshape(1, -1)
        return snake_child

    def mutation(self, snake):
        """ Apply random mutation on DNA of snake

        Args:
            snake (Snake): Snake object to alter DNA for

        Returns:
            Snake: Snake with altered DNA
        """
        # determine which gene is going to be altered
        random_values = np.random.rand(*snake.dna.shape).reshape(-1)
        random_values = random_values <= self.mutation_rate

        # add random value picked from gaussian distribution (mean=0, std=0.5) to gene
        dna = snake.dna.reshape(-1)
        mutated_dna = [gene + np.random.normal(scale=0.5) if random_value else gene
                       for gene, random_value in zip(dna, random_values)]
        snake.dna = np.asarray(mutated_dna).reshape(1, -1)
        return snake
