import yaml
from pathlib import Path

import horn_antenna

class GA:
    
    def __init__(self, run_name, settingsfile = 'settings.yaml'):
        self.run_name = run_name
        self.initialize_settings(settingsfile)
        self.generation = 0
        self.population = []
        self.fitness = []
        self.best_individual = []
        self.best_fitness = 0.0
        
    
    def initialize_settings(self, settingsfile):
        '''Initializes the settings from the settings file'''
        if Path("configs" / settingsfile).exists():
            with open(settingsfile, 'r') as file:
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

    
    def initialize_population(self, initialization=None):
        '''Initializes the population of horn antennas'''
        if initialization is None:
            for i in range(self.settings.population_size):
                self.population.append(horn_antenna())
                self.population[i].initialize()
        else:
            print("Custom initialization not implemented yet. Exiting.")
            exit(1)
            
    
    def evaluate_population(self):
        pass
    
    def selection(self):
        pass
    
    def crossover(self, parent1, parent2):
        pass
    
    def mutation(self, individual):
        pass
    
    
    def run(self):
        self.initialize_population()
        self.evaluate_population()
        self.selection()
        self.crossover()
        self.mutation()
        
        return self.best_individual, self.best_fitness