import random
import yaml
import numpy as np
import pickle
import copy
from pathlib import Path

from horn_antenna import horn_antenna
#import bicone_antenna
#import hpol_antenna

class GA:
    def __init__(self, run_name, settingsfile = 'settings.yaml'):
        self.run_name = run_name
        self.initialize_settings(settingsfile)
        self.generation = 0
        self.population = []
        self.best_individual = None
        self.best_fitness = 0.0
        self.comparison = np.array([])
        self.load_compairson()
        self.make_run_directory()
    
    
    ### Initialization ########################################################
    
    def initialize_settings(self, settingsfile):
        '''Initializes the settings from the settings file'''
        settingspath = Path(f"configs/{settingsfile}")
        if settingspath.exists():
            with open(settingspath, 'r') as file:
                settings = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print('Settings file not found. Exiting.')
            exit(1)
            
        if self.check_settings(settings):
            self.settings = settings
        else:
            print('Invalid settings. Exiting.')
            exit(1)
    
    
    def check_settings(self, settings):
        '''Checks if the settings are valid'''
        valid_settings = True
        if not(settings['crossover_rate'] + settings['mutation_rate'] +
               settings['reproduction_rate'] <= 1.0):
            valid_settings = False
        if not(settings['tournament_rate'] + settings['roulette_rate'] + 
               settings['rank_rate'] == 1.0):
            valid_settings = False
            
        return valid_settings

    
    def make_antenna(self, type, genes=None):
        '''Creates an antenna object'''
        if type == 'horn':
            return horn_antenna(genes)
        else:
            print('Invalid antenna type. Exiting.')
            exit(1)
    
    
    def initialize_population(self, initialization=None):
        '''Initializes the population of horn antennas'''
        antenna_type = self.settings['a_type']
        if initialization is None:
            for i in range(self.settings["npop"]):
                self.population.append(self.make_antenna(antenna_type))
                self.population[i].initialize()
        else:
            print("Custom initialization not implemented yet. Exiting.")
            exit(1)
    
    
    def load_compairson(self):
        '''Loads the comparison genes from the comparison file'''
        comparison_path = Path(f"comparisons/{self.settings['comparison_file']}")
        if comparison_path.exists():
            self.comparison = np.loadtxt(comparison_path)
        else:
            print('Comparison file not found. Exiting.')
            exit(1)
    
    
    def make_run_directory(self):
        '''Creates the run directory for the current run'''
        run_directory = Path("RunData") / self.run_name
        run_directory.mkdir(parents=True, exist_ok=True)
        fitness_files = run_directory / "fitness_files"
        fitness_files.mkdir(parents=True, exist_ok=True)
        gene_files = run_directory / "gene_files"
        gene_files.mkdir(parents=True, exist_ok=True)
        tracker_path = run_directory / "tracker.csv"
        with open(tracker_path, 'w') as file:
            file.write("Generation,Best Fitness,Best Individual Genes\n")
    
    ### Selection ############################################################
    
    def tournament_selection(self):
        '''selects a parent using tournament selection'''
        
        # Select tournamentsize of the population
        
        tournament_count = int(self.settings["tournament_size"] * self.settings["npop"])
        
        tournament = random.sample(self.population, tournament_count)
        
        # Find the best individual in the tournament
        best_individual = tournament[0]
        for individual in tournament:
            if individual.fitness > best_individual.fitness:
                best_individual = individual
        
        return best_individual
    
    
    def roulette_selection(self):
        '''Selects a parent using roulette selection'''
        
        # Calculate the total fitness of the population
        total_fitness = sum([individual.fitness for individual in self.population])
        
        # Calculate the probability of selection for each individual
        probabilities = [individual.fitness / total_fitness for individual in self.population]
        
        # Select an individual
        selection = random.uniform(0, 1)
        cumulative_probability = 0
        for i in range(len(self.population)):
            cumulative_probability += probabilities[i]
            if cumulative_probability > selection:
                return self.population[i]
    
    
    def rank_selection(self):
        '''Selects a parent using rank selection'''
        
        # Sort the population by fitness
        sorted_population = sorted(self.population, key=lambda x: x.fitness)
        
        # Calculate the probability of selection for each individual
        probabilities = [i / len(self.population) for i in range(1, len(self.population) + 1)]
        
        # Select an individual
        selection = random.uniform(0, 1)
        cumulative_probability = 0
        for i in range(len(self.population)):
            cumulative_probability += probabilities[i]
            if cumulative_probability > selection:
                return sorted_population[i]
    
    
    def selection(self, num_parents):
        '''Selects parents from the population'''
        parents = []
        
        for i in range(num_parents):
            selection = random.uniform(0, 1)
            if selection < self.settings["tournament_rate"]:
                parents.append(self.tournament_selection())
            elif selection < self.settings["tournament_rate"] + self.settings["roulette_rate"]:
                parents.append(self.roulette_selection())
            else:
                parents.append(self.rank_selection())
        
        return parents
        
    
    def absolute_selection(self, num_parents):
        '''Selects exactly the rate from each selection method'''
        
        no_tournament = int(num_parents * self.settings["tournament_rate"])
        no_roulette = int(num_parents * self.settings["roulette_rate"])
        no_rank = num_parents - no_tournament - no_roulette
        
        parents = []
        parents.extend([self.tournament_selection() for i in range(no_tournament)])
        parents.extend([self.roulette_selection() for i in range(no_roulette)])
        parents.extend([self.rank_selection() for i in range(no_rank)])
        
        return parents
    
    
    ### Operators ############################################################
    
    def crossover(self, parent1, parent2):
        '''Crossover two parents to create two children'''
        antenna_type = self.settings['a_type']
        
        valid_children = False
        while not valid_children:
            child1_genes = []
            child2_genes = []
            
            # Crossover genes
            for gene in range(len(parent1.genes)):
                gene_1 = parent1.genes[gene]
                gene_2 = parent2.genes[gene]
                
                coinflip = random.randint(0, 1)
                child1_genes.append(gene_1 if coinflip == 0 else gene_2)
                child2_genes.append(gene_2 if coinflip == 0 else gene_1)
                
            child1 = self.make_antenna(antenna_type, child1_genes)
            child2 = self.make_antenna(antenna_type, child2_genes)
            
            # Check if children are valid
            valid_children = child1.check_genes() and child2.check_genes()
            
        return child1, child2
                
    
    def mutation(self, individual):
        '''Mutate a randomly selected gene across a gaussian distribution'''
        chosen_gene_index = random.randint(0, len(individual.genes) - 1)
        chosen_gene = individual.genes[chosen_gene_index]
        new_indiv = copy.deepcopy(individual)
               
        valid_antenna = False
        while not valid_antenna:
            new_gene = random.gauss(chosen_gene, chosen_gene * self.settings["sigma"])
            new_indiv.genes[chosen_gene_index] = new_gene
            valid_antenna = individual.check_genes()
        
        return new_indiv
    
    
    def reproduction(individual):
        '''Asecual Reproduction'''
        return individual
    
    
    def injection(self):
        '''Injects new individuals into the population'''

        individual = self.make_antenna(self.settings['a_type'])
        individual.initialize()
        
        return individual
    
    
    ### Write/ReadFunctions ##################################################
    
    def write_population_genes(self):
        '''Write the gened of the population to the run directory'''
        filepath = Path("RunData") / self.run_name / "gene_files" / f"{self.generation}_genes.csv"
        with open (filepath, "w") as file:
            for individual in self.population:
                for gene in individual.genes[:-1]:
                    file.write(f"{gene},")
                file.write(f"{individual.genes[-1]}\n")
    
    
    def write_population_fitness(self):
        '''Write the fitness of the population to the run directory'''
        filepath = Path("RunData") / self.run_name / "fitness_files" / f"{self.generation}_fitness.csv"
        with open (filepath, "w") as file:
            for individual in self.population:
                file.write(f"{individual.fitness}\n")
    
    
    def save_population(self):
        '''save the antenna objects to a pickle file'''
        filepath = Path("RunData") / self.run_name / f"{self.generation}_population.pkl"
        with open(filepath, 'wb') as file:
            pickle.dump(self.population, file)
            
            
    def load_population(self, filepath):
        '''Load the population from a pickle file'''
        with open(filepath, 'rb') as file:
            self.population = pickle.load(file)
            
            
    def save_to_tracker(self):
        '''append the current best to the tracker file'''
        filepath = Path("RunData") / self.run_name / "tracker.csv"
        with open(filepath, 'a') as file:
            file.write(f"{self.generation},{self.best_fitness},{self.best_individual.genes}\n")
    
    
    ### Generational Methods #################################################
    
    def evaluate_population(self):
        ''' Evaluate the fitness of the entire population'''
        for individual in self.population:
            individual.evaluate_fitness(self.comparison)
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual
    
    
    def get_operator_numbers(self):
        mutation_no = int(self.settings["mutation_rate"] * self.settings["npop"])
        crossover_no = int(self.settings["crossover_rate"] * self.settings["npop"])
        reproduction_no = int(self.settings["reproduction_rate"] * self.settings["npop"])
        if mutation_no + crossover_no + reproduction_no < self.settings["npop"]:
            injection_no = self.settings["npop"] - mutation_no - crossover_no - reproduction_no
        else:
            injection_no = 0
            reproduction_no = self.settings["npop"] - mutation_no - crossover_no
            
        return mutation_no, crossover_no, reproduction_no, injection_no
    
    ### SSGA Methods #########################################################
    
    def choose_operator(self):
        '''choose an operator from the operator set of 
        REPRODUCTION, CROSSOVER, MUTATION, INJECTION'''
        # choose a random number from 0 to 1
        choice = random.uniform(0, 1)
        
        limit = self.settings["crossover_rate"]
        if choice <= limit:
            return "crossover"
        
        limit += self.settings["mutation_rate"]
        if choice <= limit:
            return "mutation"
        
        limit += self.settings["reproduction_rate"]
        if choice <= limit:
            return "reproduction"
        
        return "injection"    

    
    def create_individual(self, operator, parents):
        '''Create an individual from the operator and parents'''
        
        if operator == "crossover":
            new_indiv = self.crossover(parents[0], parents[1])
        elif operator == "mutation":
            new_indiv = self.mutation(parents[0])
        elif operator == "injection":
            new_indiv = self.injection()
        else:
            new_indiv = parents[0]
        
        if type(new_indiv) in [list, tuple]:
            new_indiv = copy.deepcopy(new_indiv[0])
        else:
            new_indiv = copy.deepcopy(new_indiv)

        return new_indiv
    
    
    def get_num_parents(self, operator):
        '''get the number of parents for a SSGA operator'''
        if operator == "crossover":
            return 2
        return 1
    
    
    def replace_individual(self, new_indiv):
        if self.settings["replacement_method"] == "random":
            index = random.randint(0, len(self.population) - 1)
            self.population[index] = new_indiv
        else:
            print("Invalid replacement method. Exiting.")
            exit(1)
    
    ### Constraint Functions #################################################
    
    
    def test_diverse(self, new_indiv):
        '''Test if an individual is identical to any
        individuals currently in the population'''
        
        unique = True
        for individual in self.population:
            if new_indiv.genes == individual.genes:
                #print("Duplicate")
                
                unique = False
                break
        
        return unique
    
    
    ### Main Loop ############################################################
    
    def advance_generation(self):
        '''Advances the generation of the population'''
        
        # check if initial generation
        if self.generation == 0:
            self.initialize_population()
            self.evaluate_population()
            self.save_population()
            self.write_population_genes()
            self.write_population_fitness()
            self.save_to_tracker()
            self.generation += 1
            return 0

        # Advance the generation
        if self.settings["steady_state"]:
            self.advance_generation_steady_state()
        else:
            self.advance_generation_generational()
        self.generation += 1
        return 0
    
    
    def advance_generation_steady_state(self):
        '''Advances the state of the GA in a steady state manner,
        creating new individuals one by one'''
        
        for i in range(self.settings["npop"]):
            
            # Create a new antenna
            valid_individual = False
            while not valid_individual:
                # Create a new individual
                
                operator = self.choose_operator()
                num_parents = self.get_num_parents(operator)
                parents = self.selection(num_parents)
                
                new_indiv = self.create_individual(operator, parents)
                
                # Check if the antenna is unique if required
                if self.settings["forced_diversity"] == True:
                    valid_individual = self.test_diverse(new_indiv)
                else:
                    valid_individual = True
                
            # Test Fitness and replace an individual
            new_indiv.evaluate_fitness(self.comparison)
            self.replace_individual(new_indiv)
            
            if new_indiv.fitness > self.best_fitness:
                self.best_fitness = new_indiv.fitness
                self.best_individual = new_indiv
            
        # Save the data
        #self.save_population()
        self.write_population_genes()
        self.write_population_fitness()
        self.save_to_tracker()
    
    
    def advance_generation_generational(self):
        new_population = []
        
        operator_nos = self.get_operator_numbers()
        
        print("Crossover")
        # Crossover
        parents = self.absolute_selection(operator_nos[0]*2)
        for i in range(operator_nos[0]):  
            print(i)
            # Create children
            valid_children = False
            while not valid_children:
                parent1_index = random.randint(0, len(parents) - 1)
                parent2_index = random.randint(0, len(parents) - 1)
                while parents[parent1_index].genes == parents[parent2_index].genes:
                    parent2_index = random.randint(0, len(parents) - 1)
                
                children = self.crossover(parents[parent1_index], parents[parent2_index])
                
                if self.settings["forced_diversity"]:
                    valid_children = (self.test_diverse(children[0]) and
                                      self.test_diverse(children[1]))
                    
                else:
                    valid_children = True
            
            # remove the parents from the list
            parents.pop(parent1_index)
            if parent1_index < parent2_index:
                parents.pop(parent2_index - 1)
            else:
                parents.pop(parent2_index)
            
            # Add the children to the new population
            new_population.extend(children)

        print("Mutation")
        # Mutation
        parents = self.absolute_selection(operator_nos[1])
        for i in range(operator_nos[1]):
            print(i)
            valid_individual = False
            while not valid_individual:
                new_indiv = self.mutation(parents[i])

                print("new_indiv", new_indiv.genes)
                print("parent", parents[i].genes)
                
                if self.settings["forced_diversity"]:
                    valid_individual = self.test_diverse(new_indiv)
                else:
                    valid_individual = True
                #rint("valid_individual", valid_individual)
                #print("new_indiv", new_indiv.genes)
            
            new_population.append(new_indiv)
        
        print("Reproduction")
        # Reproduction
        parents = self.absolute_selection(operator_nos[2])
        for i in range(operator_nos[2]):
            new_population.append(copy.deepcopy(parents[i]))
        
        print("Injection")
        # Injection
        for i in range(operator_nos[3]):
            new_population.append(self.injection())
            
        print("Evaluation")
        # Evaluate the new population
        self.population = new_population
        self.evaluate_population()
        
        # Save the data
        #self.save_population()
        self.write_population_genes()
        self.write_population_fitness()
        self.save_to_tracker()
        
                
        
    def print_stats(self):
        print(f"Generation: {self.generation}")
        print(f"Best Fitness: {self.best_fitness}")
        print(f"Best Individual: {self.best_individual.genes}")
        
        
        