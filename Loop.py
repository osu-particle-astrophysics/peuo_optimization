import argparse
from GA import GA

def parse_args():
    '''parse the run name'''
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Name of the run")
    args = parser.parse_args()
    return args


def main(args):
    '''Run the genetic algorithm'''
    genetic_algorithm = GA(args.run_name)
    genetic_algorithm.run()
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)