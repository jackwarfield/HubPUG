import numpy as np
import pandas as pd
from astropy.io import fits

def load_fits_params(filename):
    temp_fits = fits.open(filename)
    a1, a2, ang = temp_fits[0].header['POSTARG1'], temp_fits[0].header['POSTARG2'], \
      temp_fits[1].header['PA_APER']
    temp_fits.close()
    
    return a1, a2, ang

def shift_frame(df, a1, a2, ang):
    tempx = df['X'].values - a1
    tempy = df['Y'].values - a2
    tempx2 = tempx * np.cos(ang) - tempy * np.sin(ang)
    tempy2 = tempx * np.sin(ang) + tempy * np.cos(ang)

    df['X'] = tempx2
    df['Y'] = tempy2

    return df

def full_process(filename_csv, filename_fits):
    df = pd.read_csv(filename_csv)
    temp_a1, temp_a2, temp_ang = load_fits_params(filename_fits)
    temp_ang = np.pi*temp_ang/180

    df = shift_frame(df.copy(), temp_a1, temp_a2, temp_ang)
    
    return df
