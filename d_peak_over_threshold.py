# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    d_peak_over_threshold.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/11 14:32:51 by daniloceano       #+#    #+#              #
#    Updated: 2023/12/12 08:06:43 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This module, 'peak_over_threshold.py', is part of the South Atlantic waves trend analysis project, 
focusing on the application of Extreme Value Theory (EVT) in environmental and climate data analysis. 
Specifically, it implements the Peak Over Threshold (POT) method, utilizing the Generalized Pareto Distribution (GPD) 
to model and analyze exceedances over a predefined threshold.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import genpareto
import pymannkendall as mk
import os
import sys

# Constants
VARIABLE = "swh"
FIGS_DIR = f"./figures/E_{VARIABLE}/"
PROCESSED_DATA_DIR = f"./processed_data_{VARIABLE}/"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'
GREEN = '#28a745'
ORANGE = '#ffc107'
ALPHA = 0.05
POINT_LAT = -24.12
POINT_LON = -45.71
START_YEAR = "1980"
END_YEAR = "2022"

# Ensure directories exist
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_time_series(filename):
    """
    Load time series data from a CSV file.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded time series data.
    """
    try:
        data = pd.read_csv(filename, index_col=0)
        data.index = pd.to_datetime(data.index)
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)

def read_u0_from_file(filename):
    """
    Reads the threshold value u0 from a text file.

    Args:
        filename (str): Path to the text file.

    Returns:
        float: The value of u0.
    """
    try:
        with open(filename, 'r') as file:
            line = file.readline()
            return float(line.split(":")[1].strip())
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)
    except ValueError:
        print("Error in reading the u0 value from the file.")
        sys.exit(1)

def plot_histogram(data, filename, u0):
    """
    Plot and save a histogram of the data with a fitted GPD.

    Args:
        data (pandas.DataFrame): Data to plot.
        filename (str): Path to save the plot.
        u0 (float): Threshold value u0.
    """
    c, loc, scale = genpareto.fit(data - u0)
    xsi, sigma = c, scale
    min_excess, max_excess = data.min(), data.max()

    plt.figure(figsize=(16, 8))
    plt.hist(data, bins=40, density=True, color=GRAY)
    plt.xlabel("Excesses")
    plt.title("Histogram for Exceedances Over the Threshold Series $E$ and The Adjusted GPD Probability Density Function")
    
    x_fit = np.linspace(min_excess, max_excess, 500)
    h_fit = (1/sigma) * (1 + xsi * ((x_fit)/sigma))**(-1-1/xsi)
    plt.plot(x_fit, h_fit, color="red", label="GPD Probability Density Function")
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

def perform_mann_kendall_test(data, filename):
    """
    Perform Mann-Kendall test and save the results.

    Args:
        data (pandas.DataFrame): Data to test.
        filename (str): Path to save the test results.
    """
    mk_test = mk.original_test(data.values, alpha=0.05)
    mk_output = pd.Series([mk_test[0], mk_test[2]], index=["trend:", "p-value:"])
    with open(filename, 'w') as file:
        file.write(mk_output.to_string())

def plot_linear_trend(data, filename):
    """
    Plot and save the linear trend of the data.

    Args:
        data (pandas.DataFrame): Data to plot.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(16, 8))
    st1 = plt.stem(data.index, data.values)
    plt.setp(st1[0], color=RED, marker="o", markersize=2)
    plt.setp(st1[1], color=GRAY, linestyle="-")
    plt.setp(st1[2], visible=False)
    plt.xlabel("Date")
    plt.ylabel("Excesses")
    plt.title("Exceedances Over the Threshold Series $E$ and the Linear Trend Function")

    days_since_start = (data.index - data.index[0]).days.values.reshape(-1, 1)
    model = linear_model.LinearRegression().fit(days_since_start, data.values)
    y_fit = model.predict(days_since_start)

    # Extract the first (and only) coefficient from the model.coef_ array
    # Ensure that it is a scalar
    slope = model.coef_.item() if isinstance(model.coef_, np.ndarray) else model.coef_

    intercept = model.intercept_.item() if isinstance(model.intercept_, np.ndarray) else model.intercept_

    # Determine if scientific notation is needed
    use_sci_notation = abs(slope) < 0.001 or abs(intercept) < 0.001

    # Format the label accordingly
    if use_sci_notation:
        label = f"Line $y = {slope:.6f} x + {intercept:.6f}$"
    else:
        label = f"Line $y = {slope:.2f} x + {intercept:.2f}$"

    plt.plot(data.index, y_fit, color=RED, label=label)
    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

def main():
    # Define file paths
    lat, lon = np.round(float(POINT_LAT), 2), np.round(float(POINT_LON), 2)
    lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
    lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
    filename = f"E_{VARIABLE}_{lat_str}_{lon_str}"

    # Load data
    data_file = os.path.join(PROCESSED_DATA_DIR, f"E_{VARIABLE}_{lat_str}_{lon_str}_{START_YEAR}-{END_YEAR}.csv")
    E = load_time_series(data_file)

    # Read u0 from file
    u0_file = os.path.join(PROCESSED_DATA_DIR, f"u0_{VARIABLE}_{lat_str}_{lon_str}_{START_YEAR}-{END_YEAR}.txt")
    u0 = read_u0_from_file(u0_file)
    if u0 is None:
        print("Failed to read threshold u0 from file.")
        return

    # Plot histogram
    histogram_file = os.path.join(FIGS_DIR, f"histogram_{filename}.png")
    plot_histogram(E, histogram_file, u0)
    print(f"Histogram for exceedances (E) of {VARIABLE} at {lat}, {lon} saved to {histogram_file}")

    # Perform Mann-Kendall Test
    mk_test_file = os.path.join(PROCESSED_DATA_DIR, f"kendall-test_{filename}.txt")
    perform_mann_kendall_test(E, mk_test_file)
    print(f"Mann-Kendall Test for exceedances (E) of {VARIABLE} at {lat}, {lon} saved to {mk_test_file}")

    # Plot linear trend
    trend_file = os.path.join(FIGS_DIR, f"trend_{filename}.png")
    plot_linear_trend(E, trend_file)
    print(f"Linear trend for exceedances (E) of {VARIABLE} at {lat}, {lon} saved to {trend_file}")

if __name__ == "__main__":
    main()
