from GA import GA

def main():
    '''Run the genetic algorithm'''
    genetic_algorithm = GA("test14")

    for i in range(250):
        genetic_algorithm.advance_generation()
        genetic_algorithm.print_stats()
    
    
if __name__ == "__main__":
    main()