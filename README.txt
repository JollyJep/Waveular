All code has been documented on my github repo
https://github.com/JollyJep/Waveular

Code is validated to run in both python 3.9 and 3.10
----------------------------------------------------------------
Main Python Scripts (Run Surface_Waves_Core.py)
----------------------------------------------------------------

CPU_Calculation_Centre.py	#Handles all calculations if CPU if GPU Compute = False in Settings.xlsx
CUDA_Calculation_Centre.py	#Handles all calculations if GPU is GPU Compute = True in Settings.xlsx
Grid_Creation_System.py		#Converts image to particle grid
Pool_Simulation.py		#Opens pool shape png image
Settings_hub.py			#Opens Settings.xlsx and outputs setting values to rest of code
Surface_Waves_Core.py		#Main python code, runs all other connected python files and saves data. WARNING it may take time to render many particles over lots of timesteps and this script only tells you when you are finished with a real world run time of the simulation. Data will be saved anyway even if there is a crash

----------------------------------------------------------------
Graphing Script (Manually run)
----------------------------------------------------------------

Output_Centre.py			#Plots energy and position graphs, runs separatly from main simulation after main simulation has saved all files. Not automatic function, must manually run. Also hits I/O hard

----------------------------------------------------------------
Unit Tests (All manually run)
----------------------------------------------------------------

Euler-Richardson Unit Test.py		#Tests Euler-Richardson function
Force unit test.py			#Tests Hooke's law calculations
Png unit test.py			#Tests particle grid against png
Quick Shift Unit test.py		#Tests quick shift function				
Verlet Unit Test.py			#Tests velocity Verlet function
Weight Unit test.py			#Tests weight calculation

----------------------------------------------------------------
Folders
----------------------------------------------------------------

./Grid_Images	#Where all pool shape images are stored. to create custom image, use red/black pixels for fluid and white/alpha for solid surfaces (walls, islands etc)
./Output		#Where all data files are saved
./Settings		#Contains Settings.xlsx

----------------------------------------------------------------
Settings
----------------------------------------------------------------

Settings.xlsx	#Contains all initial simulation settings, default values in row 3, python used values in row 4

----------------------------------------------------------------
Dependancies (python modules)
----------------------------------------------------------------

Pillow
PyQt6
cupy		#Also need CUDA toolkit or ROCm, can be avoided if GPU Compute = False in Settings.xlsx. Need specific version that relates to CUDA/ROCm version installed https://docs.cupy.dev/en/stable/install.html
imageio
imageio-ffmpeg
matplotlib
numba
numba-scipy		#If scipy is installed
numpy
openpyxl
pandas
vispy

and all dependencies needed for all above modules
----------------------------------------------------------------
