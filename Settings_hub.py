import pandas as pd
import numpy as np


class Settings:
    """
    ----------------------------------------------------------------
    Description:
    Settings loads an excel file containing system attributes.
    ----------------------------------------------------------------
    """

    def read_settings(self):
        dataframe = pd.read_excel("./Settings/Settings.xlsx", sheet_name="Sheet1")  # Open settings excel file
        columns = dataframe.columns.tolist()    # Index columns
        self.k =float(dataframe[columns[2]][2]) # Spring constant for all springs
        self.sigma = float(dataframe[columns[3]][2]) # Surface tension coefficient
        self.c = float(dataframe[columns[4]][2]) # Damping coefficient
        self.pool_mass = float(dataframe[columns[5]][2]) # Mass of whole pool surface
        if str(dataframe[columns[6]][ 2]).lower() == "true": # Can't convert directly from string to bool, so need some logic
            self.mega_arrays = True # Use mega_arrays (arrays used to store many frames at once) needs to be true for now as minor arrays don't work
        else:
            self.mega_arrays = False
        self.VRAM = int(dataframe[columns[7]][2])   # VRAM limit in bytes
        self.repeats = int(dataframe[columns[8]][2]) # Number of repeats
        self.deltaT = float(dataframe[columns[9]][2]) # Timestep
        self.g = np.array([0, 0, float(dataframe[columns[10]][2])]) # Acceleration due to gravity
        if str(dataframe[columns[11]][
                   2]).lower() == "true":  # Can't convert directly from string to bool, so need some logic
            self.CUDA = True # If True use gpu
        else:
            self.CUDA = False