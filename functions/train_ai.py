import os
import pickle

from functions.genetic_algorithm import GeneticAlgorithm


def start_genetic_algorithm(output_folder, n_generations, population_size, survival_perc, parent_perc, mutation_perc):
    """ Start genetic algorithm and save best snake in each population and write intermediate results to file

    Args:
        output_folder (str): path to folder used to save best snake in each population and
            to write intermediate results
        n_generations (int): number of generations
        population_size (int): population size
        survival_perc (int): survival percentage. Value between 0 and 100
        parent_perc (int): parent percentage. Value between 0 and 100
        mutation_perc (int): mutation percentage. Value between 0 and 100
    """

    # scale percentage values to values between 0 and 1
    survival_perc /= 100
    parent_perc /= 100
    mutation_perc /= 100

    # Create file to save intermediate results
    file_name = 'Intermediate_results.csv'
    intermediate_results_path = "%s/%s" % (output_folder, file_name)
    # if file does not exist create one and add header
    if not os.path.exists(intermediate_results_path):
        headers = "generation,population_fitness,population_score,best_fitness,best_score\n"
        with open(intermediate_results_path, "a") as file:
            file.write(headers)

    # initialize genetic algorithm and create random population
    population = GeneticAlgorithm(population_size, survival_perc, parent_perc, mutation_perc)
    for generation in range(n_generations):

        # for each snake in the population play the game
        population.play_games(parallel=True)

        # get fitness and number of apples found for the entire population and best snake
        population_fitness = population.get_population_fitness()
        best_fitness = population.get_best_fitness()
        population_score = population.get_population_score()
        best_score = population.get_best_score()

        # get best snake and save
        best_snake = population.get_best_snake()
        file_name = "best_snake_generatie-%i_score-%i.obj" % (generation+1, best_score)
        with open("%s/%s" % (output_folder, file_name), "wb") as file:
            pickle.dump(best_snake, file=file)

        # write intermediate results to file
        content = "%i,%0.2f,%0.2f,%0.2f,%0.2f\n" % \
                  (generation+1, population_fitness, population_score, best_fitness, best_score)
        with open(intermediate_results_path, "a") as file:
            file.write(content)

        # generate new population
        population.generate_population()
