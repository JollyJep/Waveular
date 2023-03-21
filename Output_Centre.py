import time

import matplotlib.colors
import numpy as np
from numba import njit, jit, prange
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import multiprocessing as mp
import vispy as vp
import vispy.scene
from vispy.scene import visuals
import imageio
from vispy.color import color_array as ca
import colorsys




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
        plotting = False


def data_plot_system(data_hopper_2):
    plotting = True
    global data
    global scatter
    global canvas
    data = data_hopper_2.get()


    view = canvas.central_widget.add_view()
    view.camera = 'arcball'
    view.camera = vp.scene.TurntableCamera(up='z', fov=60)
    max_x = data[0][-1][0][0]
    max_y = data[0][0][-1][1]
    xax = vp.scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r',
                     font_size=32, parent=view.scene, domain=(0, max_x), minor_tick_length=5, major_tick_length=10)
    yax = vp.scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g',
                     font_size=32, parent=view.scene, domain=(0, max_y), minor_tick_length=50, major_tick_length=100)
    zax = vp.scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b',
                     font_size=32, parent=view.scene, domain=(0, 1), minor_tick_length=5, major_tick_length=10)
    zax.transform = vp.scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
    zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
    zax.transform.rotate(180, (0, 0, 1))
    scatter = vp.scene.visuals.Markers()
    view.add(scatter)
    view.camera.transform.rotate(110, (0, 0, 1))
      # or try 'arcball'
    if type(data) != bool:
        global colour_array
        colour_array = quick_colours(data)
        colour_array = ca.ColorArray(matplotlib.colors.hsv_to_rgb(colour_array))
        timer = vp.app.Timer()
        timer.connect(update)
        timer.start(0.05)

        canvas.show()
        vp.app.run()



@njit()
def quick_colours(data):
    colour_array = []
    for i, _ in enumerate(data[0]):
        for j, _ in enumerate(data[0, 0]):
            colour_array.append(((i + j) / (len(data[0]) + len(data[0, 0])), 1, 1))
    return colour_array

def update(ev):
    global data_hopper_2
    global start
    global data
    global frame
    global colour_array
    global canvas
    global writer
    if type(data) != bool:
        print(frame)
        max_x = data[0][-1][0][0]
        max_y = data[0][0][-1][1]
        divisor = np.full((len(data[0]) * len(data[0, 0]), 3), np.array([max_x, max_y, 1]))
        local_data =accelerated_formatting(data, divisor, frame)
        scatter.set_data(local_data, edge_width=0, face_color=colour_array, size=5, symbol='o')
        if frame < len(data)-1:
            frame += 1
            if start:
                im = canvas.render()
                writer.append_data(im)
        else:
            frame = 0
            data = data_hopper_2.get()
            if start and type(data) == "bool":
                start = False
                writer.close()
                print("animated")
                data = data_hopper_2.get()
            elif type(data) == "bool":
                data = data_hopper_2.get()



@njit(parallel=True)
def accelerated_formatting(data, divisor, frame):
    local_data = np.zeros((len(data[frame]) * len(data[frame][0]), 3))

    for i in prange(len(data[frame])):
        for j in prange(len(data[frame][0])):
            local_data[i * len(data[frame]) + j] = data[frame, i, j]
    return local_data / divisor


def data_formatter_host(data_hopper_1, data_hopper_2):
    plotting = True
    x = 0
    while plotting:
        x += 1
        data_mixed = data_hopper_1.get()
        if type(data_mixed) == bool:
            data_hopper_2.put(False)
            plotting = False
        else:
            data_hopper_2.put(data_mixed)
            #data = data_formatter(data_mixed)
            #print(data)
            #data_hopper_2.put(data)


if __name__ == "__main__":
    start = True
    data = None
    frame = 0
    colour_array = (1, 1, 1)
    scatter = vp.scene.visuals.Markers()
    data_hopper_1 = mp.Queue(maxsize=2)
    data_hopper_2 = mp.Queue(maxsize=2)
    data_loader_process = mp.Process(target=data_reader, args=(data_hopper_1,))
    data_formatter_process = mp.Process(target=data_formatter_host, args=(data_hopper_1, data_hopper_2,))
    #data_plotter_process = mp.Process(target=data_plot_system, args=(data_hopper_2,))
    data_loader_process.start()

    data_formatter_process.start()
    #data_plotter_process.start()
    canvas = vp.scene.SceneCanvas(keys='interactive', bgcolor='k', size=(1920, 1080))
    writer = imageio.get_writer('animation.mp4')
    data_plot_system(data_hopper_2)
