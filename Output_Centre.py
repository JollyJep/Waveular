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


def system_scanner(): # Scans the Output folder and outputs the filenames, with extensions, for data_reader to use
    path = "./Output"
    obj = os.scandir(path)
    data_modules = []
    for entry in obj:   #converts to list
        if entry.is_dir() or entry.is_file():
            data_modules.append(entry.name)

    return data_modules


def data_reader(data_export_pos, data_export_eng, ref_block_transfer, time_swap): # Used for all I/O operations, this is the reason that the output script can handle orders of magnitude more data than the RAM available
    plotting = True
    data_modules = system_scanner() # Get file names
    eng = False
    while plotting:
        for data_module in data_modules:    #For filename in list of filenames
            module = np.load("./Output/" + data_module) # Load the next file into RAM
            if "pos" in data_module:    # Sorts and sends the data depending on data type
                if eng:
                    data_export_eng.put(False)  # Queue for all energy data - Tells the energy process that data has stopped streaming and it can plot
                eng = False
                data_export_pos.put(module["arr_0"])    # Sends positional data to Vispy 3d plotting function
            elif "eng" in data_module:
                eng = True
                data_export_eng.put(module["arr_0"])    # Sends energy data to energy plotting system
            elif "time" in data_module:
                time_swap.put(module["arr_0"])  # Sends timestep information
            elif "0ref" in data_module:
                ref_block_transfer.put(ref_grid_blocker(module["arr_0"]))   # Sends pool shape

        data_export_pos.put(False)  # Indicates to Vispy function end of data streaming

        plotting = False


def data_plot_system(data_export_pos, ref_block_transfer): # Vispy 3d plotting function
    plotting = True
    global data
    global scatter
    global canvas
    data = data_export_pos.get()
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'
    view.camera = vp.scene.TurntableCamera(up='z', fov=60)
    max_x = data[0][-1][0][0] # Defines axis lengths
    max_y = data[0][0][-1][1]
    xax = vp.scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r',
                     font_size=32, parent=view.scene, domain=(0, max_x), minor_tick_length=1.5, major_tick_length=3)    # Plots x axis
    yax = vp.scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g',
                     font_size=32, parent=view.scene, domain=(0, max_y), minor_tick_length=50, major_tick_length=100)   # Plots y axis
    zax = vp.scene.Axis(pos=[[1, 0], [-1, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b',
                     font_size=32, parent=view.scene, domain=(-1, 1), minor_tick_length=1.5, major_tick_length=3)   # Plots z axis
    zax.transform = vp.scene.transforms.MatrixTransform()  # Uses an inverted x-axis
    zax.transform.rotate(90, (0, 1, 0))  # Rotate cw around y-axis
    zax.transform.rotate(180, (0, 0, 1)) # Rotates ticks
    scatter = vp.scene.visuals.Markers() # Defines empty scatter plot
    view.add(scatter)   # Adds scatter plot to output view
    view.camera.transform.rotate(110, (0, 0, 1))    # Rotates camera for better animations
    if type(data) != bool: # Detects if "End of data streaming" signal
        global colour_array
        global ref_block
        ref_block = ref_block_transfer.get()    # Not used
        colour_array = quick_colours(data)
        colour_array = ca.ColorArray(matplotlib.colors.hsv_to_rgb(colour_array)) # Converts list of colours in HSV space to Vispy colour array in RGB space. (Vispy HSV space doesn't work)
        timer = vp.app.Timer()  # Allows the plot to change as time changes
        timer.connect(update)   # Runs update function on timer
        timer.start(0.05)   # Fixes a bug in Vispy where the timer never starts if start time = 0
        canvas.show()
        vp.app.run()


@njit(parallel=True)    # Dramatically speeds up colour logic
def quick_colours(data):
    colour_array = []
    for i, _ in enumerate(data[0]): #Loops through particles
        for j, _ in enumerate(data[0, 0]):
            colour_array.append(((i + j) / (len(data[0]) + len(data[0, 0])), 1, 1)) # Creates rainbow of colours by using the angle in HSV space to define colours
    return colour_array


def update(ev): # Animation function (update isn't allowed to take variables as inputs, apart from "time")
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
        divisor = np.full((len(data[0]) * len(data[0, 0]), 3), np.array([max_x, max_y, 1])) # Formats data into a 1 by 1 by 1 cube, for easy plotting
        local_data = accelerated_formatting(data, divisor, frame)   # Calls function that provides the next timestep in an easy to use format
        scatter.set_data(local_data, face_color=colour_array, size=5, symbol='o')   # Applies new data to old particle positions
        if frame < len(data)-1: # Increases frame within the specific data file being plotted at the moment
            frame += 1
            if start:   # Saves frame in RAM for the animation, if first run
                im = canvas.render()
                writer.append_data(im)
        else:   # Preps for and accepts next data file that has been sent in the FIFO queue
            frame = 0
            data = data_export_pos.get()    # Get next data file in queue
            if isinstance(data, bool):  # Detects if the next item is the flag to state the end of the data set
                start = False
                writer.close()  # Saves and renders all frames as an mp4, using FFMPEG
                print("animated")
                data = data_export_pos.get() # Reload first data file and loop the animation


@njit(parallel=True)    # Numba vastly speeds up logic
def accelerated_formatting(data, divisor, frame):
    local_data = np.zeros((len(data[frame]) * len(data[frame][0]), 3))

    for i in prange(len(data[frame])):  # Loop through all particles
        for j in prange(len(data[frame][0])):
            local_data[i * len(data[frame]) + j] = data[frame, i, j]    # Create flat array of position 3 vectors
    return local_data / divisor  # Convert from simulation space to plotting space (1x1x1 cube)


def energy_plotter(data_export_eng, time_swap): # Uses Matplotlib to plot all energies
    plotting = True
    frames = np.zeros(1)
    start = True
    plt.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = '16'
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    while plotting:
        if start: # For first loop, collect the timestep from queue
            start = False
            deltaT = time_swap.get()
        all_energies = data_export_eng.get()    # Collect next energy data file from FIFO queue
        if not isinstance(all_energies, bool):
            kinetics_gpu = cp.sum(all_energies[0], axis=(1, 2)) # Energies from 3 vectors to scalars
            gpe_gpu = cp.sum(all_energies[1], axis=(1, 2))
            epe_gpu = cp.sum(all_energies[2], axis=(1, 2))
            frames = np.linspace(max(frames), (len(kinetics_gpu) + max(frames)-1), len(kinetics_gpu))   # The current set of times to plot on the x axis
            if frames[0] == 0:  # Plot first data file with labels
                ax.plot(frames * deltaT, cp.asnumpy(kinetics_gpu), label="Kinetic Energy", color="red", ls=(0, (3, 10, 1, 10, 1, 10)))
                ax.plot(frames * deltaT, cp.asnumpy(gpe_gpu), label="Gravitational Potential Energy", color="lightblue", ls="--")
                ax.plot(frames * deltaT, cp.asnumpy(epe_gpu), label="Elastic Potential Energy", color="darkblue", ls="dotted")
                ax.plot(frames * deltaT, cp.asnumpy(epe_gpu + gpe_gpu), label="Potential Energy", color="blue", ls="dashdot")
                ax.plot(frames * deltaT, cp.asnumpy(kinetics_gpu + gpe_gpu + epe_gpu), label="Total Energy", color="purple")
            else:   # In blocks of data, plot each data file
                ax.plot(frames * deltaT, cp.asnumpy(kinetics_gpu), color="red", ls=(0, (3, 10, 1, 10, 1, 10)))
                ax.plot(frames * deltaT, cp.asnumpy(gpe_gpu), color="lightblue", ls="--")
                ax.plot(frames * deltaT, cp.asnumpy(epe_gpu), color="darkblue", linestyle="dotted")
                ax.plot(frames * deltaT, cp.asnumpy(epe_gpu + gpe_gpu), color="blue", ls="dashdot")
                ax.plot(frames * deltaT, cp.asnumpy(kinetics_gpu + gpe_gpu + epe_gpu), color="purple")
        else:   # At the end of all data streaming, launch plot
            plt.legend()
            plt.ylabel("Energy/ J", fontsize=20)
            plt.xlabel("Time/ s", fontsize=20)
            plt.show()
            plotting = False



def ref_grid_blocker(ref_grid): # Not used
    return np.place(ref_grid, np.invert(ref_grid), np.nan)


if __name__ == "__main__": # Multiprocessing protection
    start = True    # Define global variables (update isn't allowed to take variables as inputs)
    data = None
    frame = 0
    colour_array = (1, 1, 1)
    scatter = vp.scene.visuals.Markers()
    data_export_pos = mp.Queue(maxsize=4)   # FIFO data queues used for streaming the data from disk to functions. Maxsize limits maximum RAM usage at any one time.
    data_export_eng = mp.Queue(maxsize=4)
    ref_block_transfer = mp.Queue(maxsize=1)
    time_swap = mp.Queue(maxsize=1)
    ref_block = 0
    data_loader_process = mp.Process(target=data_reader, args=(data_export_pos, data_export_eng, ref_block_transfer, time_swap, ))  # Define processes that run each function. Each process has its own GIL and hence streaming doesn't stall the plotting processes.
    plot_energy_process = mp.Process(target=energy_plotter, args=(data_export_eng, time_swap, ))
    data_loader_process.start()
    plot_energy_process.start()
    canvas = vp.scene.SceneCanvas(keys='interactive', bgcolor='k', size=(1920, 1080))   # Create Vispy window, note size does not work, window renders at whatever pexel size it feels like. Vispy is not very stable, but necessary for the number of particles used, due to GPU acceleration
    writer = imageio.get_writer('animation.mp4', fps=30, quality=10)    # Define the output animation and codec, to be rendered using FFMPEG
    data_plot_system(data_export_pos, ref_block_transfer)   # Launch Vispy plotting, Vispy only works when on the main process
