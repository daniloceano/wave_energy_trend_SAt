# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    get_unseazon_series.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/08 15:54:34 by daniloceano       #+#    #+#              #
#    Updated: 2023/12/11 12:39:11 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script performs analysis and visualization of time series data, specifically for deseasonalizing
and detrending a given time series. The script follows methodologies outlined
in the research article "The increase in intensity and frequency of surface air temperature extremes
throughout the western South Atlantic coast". It includes functionalities to load data, plot and save
various types of visualizations, calculate moving statistics, and perform deseasonalization of time series data.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
VARIABLE = "swh"
FIGS_DIR = "./figures/"
PROCESSED_DATA_DIR = f"./processed_data_{VARIABLE}/"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'

# Ensure directories exist
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
for subdir in ["X", "Z", "W"]:
    os.makedirs(os.path.join(FIGS_DIR, f"{subdir}_{VARIABLE}"), exist_ok=True)

def configure_matplotlib():
    """
    Configures global settings for Matplotlib visualizations.

    This function sets a default style and customizes various parameters like color cycles,
    font size, and the visibility of certain axes spines to enhance the visual appeal of plots.
    """
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[GRAY])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

def load_data(file_pattern):
    """
    Load data from multiple netCDF files matching a given pattern.

    Args:
        file_pattern (str): A file pattern to match netCDF files, typically with a wildcard character.

    Returns:
        xarray.Dataset: A dataset containing data from the matched netCDF files. The time variable
                        is converted to a pandas datetime object for easier manipulation.
    """
    ds = xr.open_mfdataset(file_pattern)
    ds['time'] = pd.to_datetime(ds.variables["time"])
    return ds

def plot_and_save(data, plot_type, title, xlabel, ylabel, filename, subdirectory="", time=None, avg=None, std=None, **kwargs):
    """
    Generates and saves a plot based on the given data and plot type.

    This function supports various plot types including histograms, time series, and boxplots.
    It now includes the functionality to save plots in specific subdirectories based on the data type (e.g., X, Z, W).

    Args:
        data (numpy.ndarray): The data to be plotted.
        plot_type (str): Type of the plot (e.g., 'histogram', 'time_series', 'boxplot').
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        filename (str): The filename to save the plot as.
        subdirectory (str): The subdirectory within the main figures directory where the plot will be saved.
                            This is used to organize plots by data type (e.g., 'X', 'Z', 'W').
        time (numpy.ndarray, optional): Time data for time series plots.
        avg (numpy.ndarray, optional): Data for the moving average line in time series plots.
        std (numpy.ndarray, optional): Data for the standard deviation in time series plots.
        **kwargs: Additional keyword arguments for specific plot customizations.

    Raises:
        ValueError: If an invalid plot type is specified.
    """
    plt.figure(figsize=(16, 8))
    
    if plot_type == 'histogram':
        mean = np.nanmean(data)
        plt.hist(np.ravel(data), bins=kwargs.get('bins', 50), color=GRAY)
        plt.axvline(np.nanmean(data), color=RED, label=f'Mean: {mean:.2f}')
        plt.legend()
    elif plot_type == 'time_series':
        plt.plot(time, data, color=GRAY, label="Original Series")
        if avg is not None:
            plt.plot(time, avg, color=RED, label="Moving Average")
        if std is not None:
            std_upper, std_lower = avg + std, avg - std
            plt.fill_between(time, std_lower, std_upper, color=BLUE, alpha=0.3,
                            zorder=100, label="Standard Deviation")
        if avg is not None or std is not None:
            plt.legend()
        plt.xlim(data.index[0], data.index[-1])
    elif plot_type == 'boxplot':
        months = kwargs.get('months')
        bp1 = plt.boxplot([data[data.index.month == m] for m in range(1, 13)], labels=months)
        plt.setp(bp1["medians"], color = RED, linewidth=3)
        plt.setp(bp1["boxes"], color = GRAY)
        plt.setp(bp1["whiskers"], color = GRAY, linestyle = "--")
        plt.setp(bp1["fliers"], color = GRAY, marker = "+")
    else:
        raise ValueError("Invalid plot type specified.")
    
    filename_path = os.path.join(FIGS_DIR, subdirectory, f"{plot_type}_{filename}.png")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.savefig(filename_path)
    plt.close()

def calculate_moving_stats(data, window_size):
    """
    Calculates moving statistics (mean and standard deviation) for the given data.

    Args:
        data (pandas.Series): The data series on which to calculate the moving statistics.
        window_size (int): The size of the moving window.

    Returns:
        tuple: A tuple containing two pandas.Series for moving average and moving standard deviation.
    """
    return data.rolling(center=False, window=window_size).mean(), data.rolling(center=False, window=window_size).std()

def deseasonalize_data(data):
    """
    Deseasonalizes the given time series data.

    This function removes the seasonal component of the data, making it more suitable for
    certain types of statistical analysis.

    Args:
        data (pandas.Series): The data series to be deseasonalized.
        index (pandas.Index): The index of the data series.

    Returns:
        pandas.Series: The deseasonalized data series.
    """
    W_list = []
    for m in range(1, 13):
        month_values = data[data.index.month == m]
        month_mean = data[data.index.month == m].mean()
        month_std = data[data.index.month == m].std()
        W_list.append((month_values - month_mean) / month_std)
    W = pd.concat(W_list)
    return W.dropna().sort_index()

def main():
    configure_matplotlib()

    # Load data
    ds = load_data("Hs*nc")
    lat, lon = np.round(float(ds.latitude), 2), np.round(float(ds.longitude), 2)
    lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
    lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
    X = ds.squeeze(['longitude', 'latitude']).to_dataframe().drop(columns=["latitude", "longitude"])
    datetime = pd.to_datetime(X.index)

    # Parameters
    variable_long_name = ds.variables[VARIABLE].attrs["long_name"]
    plot_title = f"{variable_long_name} ({datetime[0].year} to {datetime[-1].year})"
    filename = f"{VARIABLE.upper()}_{lat_str}_{lon_str}"

    # Compute values for main variable and plot
    data = X[VARIABLE]
    plot_and_save(data, 'histogram', plot_title, VARIABLE.upper(), "Frequency", filename, subdirectory=f"X_{VARIABLE}")
    plot_and_save(data, 'boxplot', plot_title, "month", VARIABLE.upper(), filename, subdirectory=f"X_{VARIABLE}")
    plot_and_save(data, 'time_series', plot_title, "", VARIABLE.upper(), filename, subdirectory=f"X_{VARIABLE}", time=datetime)
    print(f"Basic statistics plots for {VARIABLE} at {lat}, {lon} created")

    # Calculate and plot rolling statistics
    moving_avg, moving_std = calculate_moving_stats(X[VARIABLE], 8760)
    filename_rolling = f"rolling_{filename}"
    plot_and_save(X[VARIABLE], 'time_series', plot_title, "", f"{VARIABLE.upper()} (m)", filename_rolling,  
                  subdirectory=f"X_{VARIABLE}", time=datetime, avg=moving_avg, std=moving_std)
    print(f"Rolling statistics plots for {VARIABLE} at {lat}, {lon} created")

    # Detrending data
    Z = X[VARIABLE] - moving_avg
    Z = Z.dropna()

    # Plot detrended data
    detrended_filename = f"Z_{filename}"
    deseasonalized_title = f"Time Series Detrended by Subtraction of Its Moving Average ($Z$) \n{plot_title}"
    plot_and_save(Z, 'time_series', deseasonalized_title, "Date", f"Translated {VARIABLE.upper()}",
                detrended_filename, subdirectory=f"Z_{VARIABLE}", time=Z.index)
    print(f"Detrended (Z) plots for {VARIABLE} at {lat}, {lon} created")

    # Deseazonalizing the time series
    W = deseasonalize_data(Z)
    W.rename('normalized deviation', inplace=True)

    # Plot deseazonalized time series
    deseasonalized_filename = f"W_{filename}"
    deseasonalized_title = f"Deseazonalized ($W$) \n{plot_title}"
    plot_and_save(W, 'histogram', deseasonalized_title, "Normalized Deviations", "Frequency", deseasonalized_filename, subdirectory=f"W_{VARIABLE}")
    plot_and_save(W, 'boxplot', deseasonalized_title, "Month", "Normalized Deviations", deseasonalized_filename, subdirectory=f"W_{VARIABLE}")
    plot_and_save(W, 'time_series', deseasonalized_title, "Date", "Normalized Deviations",
                  deseasonalized_filename, time=W.index, subdirectory=f"W_{VARIABLE}")
    print(f"Deseasonalized (W) plots for {VARIABLE} at {lat}, {lon} created")

    # Save processed data
    W.to_csv(os.path.join(PROCESSED_DATA_DIR, f"W_{VARIABLE}_{lat_str}_{lon_str}_{datetime[0].year}-{datetime[-1].year}.csv"))
    print(f"Deseasonalized (W) series saved to {os.path.join(PROCESSED_DATA_DIR, f'W_{VARIABLE}_{lat_str}_{lon_str}_{datetime[0].year}-{datetime[-1].year}.csv')}")

if __name__ == "__main__":
    main()
