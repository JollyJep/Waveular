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
import cupy as cp


def system_scanner():
    path = "./Output"
    obj = os.scandir(path)
    data_modules = []
    for entry in obj:
        if entry.is_dir() or entry.is_file():
            data_modules.append(entry.name)

    return data_modules


def data_reader(data_export_pos, data_export_eng):
    plotting = True
    data_modules = system_scanner()
    while plotting:
        for data_module in data_modules:
            module = np.load("./Output/" + data_module)
            if "pos" in data_module:
                data_export_eng.put(False)
                data_export_pos.put(module["arr_0"])
            elif "eng" in data_module:
                data_export_eng.put(module["arr_0"])
        data_export_pos.put(False)

        plotting = False


def data_plot_system(data_export_pos):
    plotting = True
    global data
    global scatter
    global canvas
    data = data_export_pos.get()
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'
    view.camera = vp.scene.TurntableCamera(up='z', fov=60)
    max_x = data[0][-1][0][0]
    max_y = data[0][0][-1][1]
    xax = vp.scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r',
                     font_size=32, parent=view.scene, domain=(0, max_x), minor_tick_length=1.5, major_tick_length=3)
    yax = vp.scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g',
                     font_size=32, parent=view.scene, domain=(0, max_y), minor_tick_length=50, major_tick_length=100)
    zax = vp.scene.Axis(pos=[[1, 0], [-1, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b',
                     font_size=32, parent=view.scene, domain=(-1, 1), minor_tick_length=1.5, major_tick_length=3)
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
    global data_export_pos
    global start
    global data
    global frame
    global colour_array
    global canvas
    global writer
    if type(data) != bool:
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
            data = data_export_pos.get()
            if isinstance(data, bool):
                start = False
                writer.close()
                print("animated")
                data = data_export_pos.get()


@njit(parallel=True)
def accelerated_formatting(data, divisor, frame):
    local_data = np.zeros((len(data[frame]) * len(data[frame][0]), 3))

    for i in prange(len(data[frame])):
        for j in prange(len(data[frame][0])):
            local_data[i * len(data[frame]) + j] = data[frame, i, j]
    return local_data / divisor

def energy_plotter(data_export_eng):
    plotting = True
    frames = np.zeros(1)
    while plotting:
        all_energies = data_export_eng.get()
        if not isinstance(all_energies, bool):
            kinetics_gpu = cp.sum(all_energies[0], axis=(1,2))
            gpe_gpu = cp.sum(all_energies[1], axis=(1,2))
            epe_gpu = cp.sum(all_energies[2], axis=(1,2))
            frames = np.linspace(max(frames), len(kinetics_gpu) + max(frames), len(kinetics_gpu))
            if frames[0] == 0:
                plt.plot(frames, cp.asnumpy(kinetics_gpu), label="Kinetic Energy", color="red")
                plt.plot(frames, cp.asnumpy(gpe_gpu), label="Gravitational Potential Energy", color="lightblue")
                plt.plot(frames, cp.asnumpy(epe_gpu), label="Elastic Potential Energy", color="darkblue")
                plt.plot(frames, cp.asnumpy(epe_gpu + gpe_gpu), label="Potential Energy", color="blue")
                plt.plot(frames, cp.asnumpy(kinetics_gpu + gpe_gpu + epe_gpu), label="Total Energy", color="purple")
            else:
                plt.plot(frames, cp.asnumpy(kinetics_gpu), color="red")
                plt.plot(frames, cp.asnumpy(gpe_gpu), color="lightblue")
                plt.plot(frames, cp.asnumpy(epe_gpu), color="darkblue")
                plt.plot(frames, cp.asnumpy(epe_gpu + gpe_gpu), color="blue")
                plt.plot(frames, cp.asnumpy(kinetics_gpu + gpe_gpu + epe_gpu), color="purple")
        else:
            plt.legend()
            plt.show()
            plotting = False



if __name__ == "__main__":
    start = True
    data = None
    frame = 0
    colour_array = (1, 1, 1)
    scatter = vp.scene.visuals.Markers()
    data_export_pos = mp.Queue(maxsize=2)
    data_export_eng = mp.Queue(maxsize=4)
    data_loader_process = mp.Process(target=data_reader, args=(data_export_pos, data_export_eng, ))
    plot_energy_process = mp.Process(target= energy_plotter, args=(data_export_eng, ))
    data_loader_process.start()
    plot_energy_process.start()
    #data_plotter_process.start()
    canvas = vp.scene.SceneCanvas(keys='interactive', bgcolor='k', size=(1920, 1080))
    writer = imageio.get_writer('animation.mp4', fps=30, quality=10)
    data_plot_system(data_export_pos)
