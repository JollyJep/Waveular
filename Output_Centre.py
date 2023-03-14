import numpy as np
import numba
from numba import njit
import scipy
import pandas as pd
from matplotlib import pyplot as plt
import os
import threading as th
import multiprocessing as mp
import queue


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



#def data_plot_system():
#    global data_hopper_2
#    global plotting
#    while plotting:


def data_formatter_host(data_hopper_1, data_hopper_2):
    plotting = True
    while plotting:
        data_mixed = data_hopper_1.get()
        print(type(data_mixed))
        if type(data_mixed) == bool:
            data_hopper_2.put(False)
        else:
            data = data_formatter(data_mixed)
            data_hopper_2.put(data)

#@njit()
def data_formatter(data_mixed):
    data_z = data_mixed[:,:,:,0]
    print(data_z[0][500, 500])
    return data_z




if __name__ == "__main__":
    data_hopper_1 = queue.Queue(maxsize=4)
    data_hopper_2 = queue.Queue(maxsize=4)
    data_loader_thread = th.Thread(target=data_reader, args=(data_hopper_1,))
    data_formatter_process = mp.Process(target=data_formatter_host, args=(data_hopper_1, data_hopper_2,))
    data_loader_thread.start()
    data_formatter_host(data_hopper_1, data_hopper_2)
    #data_formatter_process.start()

