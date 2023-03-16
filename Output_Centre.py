import time
import numpy as np
import numba
from numba import njit
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly as px
import os
import threading as th
import multiprocessing as mp
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


def system_scanner():
    path = "./Output"
    obj = os.scandir(path)
    data_modules = []
    for entry in obj:
        if entry.is_dir() or entry.is_file():
            data_modules.append(entry.name)
    return data_modules


def data_reader(data_hopper_1):
    plotting = True
    data_modules = system_scanner()
    while plotting:
        for data_module in data_modules:
            module = np.load("./Output/" + data_module)
            data_hopper_1.put(module["arr_0"])
        data_hopper_1.put(False)


def data_plot_system(data_hopper_2):

    plotting = True
    while plotting:
        data = data_hopper_2.get()
        if type(data) != bool:
            for x in range(7):
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                ax.set(xlim3d=(0, 11), xlabel='X')
                ax.set(ylim3d=(0, 11), ylabel='Y')
                ax.set(zlim3d=(-0.2, 0.2), zlabel='Z')
                lines = [ax.scatter(data[x, i, j, 0], data[x, i, j, 1], data[x, i, j, 2]) for i in range(10) for j in range(10)]
                #ani = animation.FuncAnimation(fig, update_lines, len(data), fargs=(data, lines, ax), interval=1)
                #plt.show()

                plt.savefig(str(x)+".png")
                plt.cla()
            print("done")
            exit()


def update_lines(frame, data, lines, ax):
    print(np.shape(data), data[frame][0,0,0])
    for i in range(10):
        for j in range(10):
            x,y,z = data[frame][i, j, 0:1], data[frame][ i, j, 1:2], data[frame][ i, j, 2:]
            lines[i * 10 + j]._offsets3d(x,y,z)
    return lines



def data_formatter_host(data_hopper_1, data_hopper_2):
    plotting = True
    x=0
    while plotting:
        x += 1
        data_mixed = data_hopper_1.get()
        if type(data_mixed) == bool:
            data_hopper_2.put(False)
        else:
            data_hopper_2.put(data_mixed)
            #data = data_formatter(data_mixed)
            #print(data)
            #data_hopper_2.put(data)

@njit()
def data_formatter(data_mixed):
    x = []
    y = []
    z = []
    for index, time_step in enumerate(data_mixed):
        loc_x = []
        loc_y = []
        loc_z = []
        for i in data_mixed[index]:
            for j in i:
                loc_x.append(j[0])
                loc_y.append(j[1])
                loc_z.append(j[2])
        x.append(loc_x)
        y.append(loc_y)
        z.append(loc_z)
    return x, y, z




if __name__ == "__main__":


    data_hopper_1 = mp.Queue(maxsize=2)
    data_hopper_2 = mp.Queue(maxsize=2)
    data_loader_process = mp.Process(target=data_reader, args=(data_hopper_1,))
    data_formatter_process = mp.Process(target=data_formatter_host, args=(data_hopper_1, data_hopper_2,))
    data_plotter_process = mp.Process(target=data_plot_system, args=(data_hopper_2,))
    data_loader_process.start()

    data_formatter_process.start()
    data_plotter_process.start()
    data_formatter_host(data_hopper_1, data_hopper_2)
