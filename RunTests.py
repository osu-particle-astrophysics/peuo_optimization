import argparse
from GA import GA

def main():
    '''Run 100 runs of ssga and genga from the same initialization file.'''
    
    run_names = ["new_genga_trr", "new_genga_t", "new_ssga_t", "new_ssga_t_fd", "new_genga_trr__fd", "new_genga_t_fd"]
    settingsfiles = ["genga_no_fd.yaml", "genga_t_no_fd.yaml", "ssga_no_fd.yaml", "ssga.yaml", "genga.yaml", "genga_t.yaml"]
    
    for i in range(1, 101):
        for j, run_name in enumerate(run_names):
            ga = GA(f"{run_name}_{i}", settingsfile=settingsfiles[j], initialization=f"init_{i}")
            ga.run()
    
    
if __name__ == "__main__":
    main()
    