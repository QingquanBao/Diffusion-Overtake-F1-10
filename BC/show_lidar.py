import pickle
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

def load_lidar_data(fn):
    csv = pd.read_csv(fn, skiprows=1, header=None).iloc[:, :-1]
    r = csv.to_numpy()[:,2:]

    t = np.arange(0, r.shape[0], 1)*0.1
    r = [ {'t': t[i], 'scan': r[i]} for i in range(r.shape[0]) ]
    return r

# these are some functions to visualize lidar data
def show_lidar(d):
    # angles of each lidar ray are in a field of view [-135, 135] degree about
    # the optical axis
    th = np.arange(0,270.25,0.25)*np.pi/180.0

    plt.figure(1); plt.clf();
    ax = plt.subplot(111, projection='polar')
    try:
        for i in range(0, len(d), 10):
            d[i]['scan'][d[i]['scan'] > 30] = 30

            ax.clear()
            ax.plot(th, d[i]['scan'])
            ax.plot(th, d[i]['scan'], 'r.')
            ax.set_rmax(10)
            ax.set_rticks([0.5,1,1.5,2])
            ax.set_rlabel_position(-22.5)
            ax.grid(True)
            ax.set_title('Lidar scans [%d]: %2.3f [sec]'%(i, d[i]['t']))

            # set 135 degrees to be on top
            ax.set_theta_offset(-np.pi / 4)
            #ax.set_theta_direction(-1)

            plt.draw()
            plt.pause(1e-3)

    except KeyboardInterrupt:
        plt.close(1)


if __name__ == "__main__":
    from pathlib import Path
    import os

    parent_path = Path('./') #Path(__file__).parent
    folder = Path.joinpath(parent_path, 'data', 'Gulf_dataset')

    check_name = re.compile('^ML')
    datasets = []
    for filename in sorted(os.listdir(folder)):
        files = os.path.join(folder, filename)
        if re.match(check_name, filename) and os.path.isfile(files):
            datasets.append(files)
        
    file_idx = 3

    lidar_path = datasets[file_idx]
    lidars = load_lidar_data(lidar_path)
    show_lidar(lidars)