import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    '''Plot the 10 trials of genga and ssga'''

    #plot_names = ["GENGA TTR", "GENGA TTR FD", "GENGA T", "GENGA T FD", "SSGA T", "SSGA T FD"]
    #run_names = ["genga_no_fd_run", "genga_run", "genga_t_no_fd_run", "genga_run_only_t", "ssga_no_fd_run", "ssga_run"]
    
    run_names = ["new_genga_trr", "new_genga_trr__fd", "new_genga_t", "new_genga_t_fd", "new_ssga_t", "new_ssga_t_fd"]
    plot_names = ["GENGA TTR", "GENGA TTR FD", "GENGA T", "GENGA T FD", "SSGA T", "SSGA T FD"]
    plot_averages(run_names, plot_names, 100)


def plot_all_lines():
    plt.clf()
    
    rundir = Path("RunData")
    for i in range(1, 100):
        # Only read in the second column
        ssga = np.loadtxt(rundir / f"ssga_run_{i}" / "tracker.csv", delimiter=',', usecols=[1], skiprows=1)
        genga = np.loadtxt(rundir / f"genga_run_{i}" / "tracker.csv", delimiter=',', usecols=[1], skiprows=1)
        
        
        plt.plot(ssga, label="SSGA", color='r', alpha=0.5)
        plt.plot(genga, label="GENGA", color='b', alpha=0.5)
        
    plt.xlabel("Generation")
    plt.ylabel("Maximum Fitness")
    plt.title("Comparison of SSGA and GENGA")
    
    # Add legend, red for ssga, blue for genga
    
    red_line = plt.Line2D([0], [0], color='r', label='SSGA')
    blue_line = plt.Line2D([0], [0], color='b', label='GENGA')
    plt.legend(handles=[red_line, blue_line])
    
    plt.savefig("comparison.png")


def plot_averages(run_name_list, plot_name_list, count):
    '''Plot the averages of the runs in run_name_list'''
    
    # Clear Plot
    plt.clf()
    rundir = Path("RunData")
    colors = ['r', 'b', 'g', 'y', 'm', 'c']
    color_lines = []
    
    plt.figure(figsize=(8, 6))
    
    for j, run_name in enumerate(run_name_list):
        avg = np.zeros(100)
        
        for i in range(1, count + 1):
            run = np.loadtxt(rundir / f"{run_name}_{i}" / "tracker.csv",
                             delimiter=',', usecols=[1], skiprows=1)
            avg += run
            
        avg /= count
        
        plt.plot(avg, label=run_name, color=colors[j])
        
        
        color_lines.append(plt.Line2D([0], [0], color=colors[j],
                                        label=f"Average of {count} Trials of {plot_name_list[j]}"))
        
    
    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Maximum Fitness", fontsize=16)
    
    plt.title("Comparison of PUEO Genetic Algorithms", fontsize=20)
    
    plt.legend(handles=color_lines, fontsize=11)
    
    plt.ylim(0.75, 1.0)
    
    save_name = "all_comparisons_new"
    
    plt.text(100, 0.86, "T = Only Tournament Selection\n"
                   "TRR = All Selection Methods\n"
                   "FD = Forced Diversity", zorder=10, color='black',
                     fontsize=9, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
    
    
    plt.savefig(f"{save_name}.png")



if __name__ == '__main__':
    main()