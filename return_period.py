from get_exceedance_series import load_time_series as load_W
from peak_over_threshold import read_u0_from_file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thresholdmodeling import thresh_modeling

import os

# Constants
VARIABLE = "swh"
FIGS_DIR = f"./figures/W_{VARIABLE}/"
PROCESSED_DATA_DIR = f"./processed_data_{VARIABLE}/"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'
ALPHA = 0.05
POINT_LAT = -24.12
POINT_LON = -45.71
START_YEAR = "1980"
END_YEAR = "2022"

# Import detrending series
print(f"Performing return period analysis for {VARIABLE} at {POINT_LAT}, {POINT_LON}")
lat, lon = np.round(float(POINT_LAT), 2), np.round(float(POINT_LON), 2)
lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
W = load_W(VARIABLE, lat_str, lon_str, START_YEAR, END_YEAR)
filename = f"{VARIABLE.upper()}_{lat_str}_{lon_str}"

# constructing the maximum daily normalized deviation time series from the time series W
W_max_daily = W.resample('D').max()
W_max_daily = W_max_daily.rename(columns={'normalized deviation': 'maximum daily normalized deviation'})

# Read u0 from file
u0_file = os.path.join(PROCESSED_DATA_DIR, f"u0_{VARIABLE}_{lat_str}_{lon_str}_{START_YEAR}-{END_YEAR}.txt")
u0 = read_u0_from_file(u0_file)

#### N–year return level for the time series W_max_daily with a confidence level of 5%
block_size = 365 # number of days in the block/number of observations per year
N = 5 # number of years of interest
return_period = N * block_size # N–year return period

plt.close("all")
return_level_filename = os.path.join(FIGS_DIR, f"return_level_W_{filename}.png")
fig = thresh_modeling.return_value(W_max_daily['maximum daily normalized deviation'], u0, 0.05, block_size, return_period, "mle")
plt.savefig(return_level_filename)
print(f"Return level of W {VARIABLE} at {lat}, {lon} saved to {return_level_filename}")
