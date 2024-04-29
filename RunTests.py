import argparse
from GA import GA

def main():
    '''Run 100 runs of ssga and genga from the same initialization file.'''
    for i in range(1, 101):
        print(i)
        ga = GA(f"genga_t_no_fd_run_{i}", settingsfile="genga_t_no_fd.yaml", initialization=f"init_{i}")
        ga.run()
        
    print("SSGA Done")
        
    for i in range(1, 101):
        print(i)
        ga = GA(f"genga_no_fd_run_{i}", settingsfile="genga_no_fd.yaml", initialization=f"init_{i}")
        ga.run()

    print("GENGA Done")
    
    
if __name__ == "__main__":
    main()
    