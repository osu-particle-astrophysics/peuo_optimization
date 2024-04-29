import random
import numpy as np
from pathlib import Path

class horn_antenna:
    
    def __init__(self, genes=None):
        self.genes = genes
        self.fitness = 0.0
        self.true_fitness = 0.0
        self.MAX_S = 50.0
        self.MAX_H = 175.0
        self.MIN_H = 75.0
        
    
    def initialize(self):
        '''Initializes the genes of the horn antenna. 
        The genes are: [side_length, height, x_0, y_0, z_f, y_f, 
        beta, trpzd_length, trpzd_height]'''
        
        valid_design = False
        while valid_design == False:
            # Generate random values
            side_length = random.uniform(0.0, self.MAX_S)
            height = random.uniform(self.MIN_H, self.MAX_H)
            x_0 = random.uniform(0.0, side_length)
            y_0 = random.uniform(0.0, x_0)
            z_f = random.uniform(0.0, height)
            y_f = random.uniform(0.0, z_f)
            beta = random.uniform((4.0/30.0) * z_f, 7.0 * z_f) / 100.0
            trpzd_length = random.uniform(0.0, y_0)
            trpzd_height = random.uniform(0.0, x_0)
            
            self.genes = [side_length, height, 
                          x_0, y_0, z_f, y_f, 
                          beta, trpzd_length, 
                          trpzd_height]
            
            valid_design = self.check_genes()
        
        
    
    def check_genes(self):
        '''Checks if the genes are valid. Returns True if the genes are invalid'''
        
        # Load genes
        (side_length, height, x_0, y_0, z_f, y_f, 
        beta, trpzd_length, trpzd_height) = self.genes

        # Variables
        valid_design = False
        x_f = side_length
        
        # Calculate trapezoid intersection value
        trpzd_intersect = (trpzd_length * x_0 + trpzd_height * y_0 -
            x_0 * y_0) / (trpzd_height + trpzd_length - y_0)
        
        # Run checks
        if (not(0 <= side_length <= self.MAX_S) or 
            not(self.MIN_H <= height <= self.MAX_H)):
            valid_design = False
        elif not(0 <= x_0 <= x_f):
            valid_design = False
        elif not(0 <= y_0 <= min(z_f, x_0)):
            valid_design = False
        elif not(0 <= y_f <= z_f):
            valid_design = False
        elif not(0 <= z_f <= height):
            valid_design = False
            
        # Check if trapezoids touch
        elif x_0 - trpzd_height < trpzd_intersect < x_0:
            valid_design = False
        elif ((4.0 / 30.0) * z_f > (beta * 100) or 
              (7 * z_f) < (beta * 100) or beta > 2):
            valid_design = False
        else:
            valid_design = True

        return valid_design
    
    
    def evaluate_fitness(self, compairson):
        '''Calculate the euclidean distance between the genes
        of the horn antenna and the comparison genes'''
        
        euclidean_distance = 0.0
        gene_count = len(self.genes)
        for i in range(gene_count):
            numerator = (self.genes[i] - compairson[i]) ** 2
            denominator = (self.genes[i] + compairson[i]) ** 2
            euclidean_distance += numerator / denominator
            
        normalized_distance = euclidean_distance / (2 * gene_count)
        
        normalized_distance = normalized_distance ** 0.5
        
        self.fitness = 1.0 - normalized_distance
    
    
    def save_as_comparison(self, filename):
        '''save the current genes as a comparison file'''
        filepath = Path(f"comparisons/{filename}.txt")
        np.savetxt(filepath, self.genes)
    
    
    def __str__(self) -> str:
        '''Returns a string representation of the antenna's genes'''
        return str(self.genes)
        