# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    stationarity_analysis.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/08 16:56:42 by daniloceano       #+#    #+#              #
#    Updated: 2023/12/11 17:26:18 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This module is designed for advanced statistical analysis and visualization of time series data, 
with specific applications in the research article "The increase in intensity and frequency of surface 
air temperature extremes throughout the western South Atlantic coast". It includes functionalities 
for creating lag plots, analyzing autocorrelation, performing stationarity tests like the Augmented 
Dickey-Fuller test, and generating hourly boxplots. The module supports the analysis of deseasonalized 
time series data. The methods implemented here, such as the ADF test and the Ljung-Box Test, 
are crucial for ensuring the reliability and accuracy of the time series analysis as outlined in the research.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import lag_plot

# Constants
VARIABLE = "swh"
FIGS_DIR = "./figures/W_swh/"
PROCESSED_DATA_DIR = f"./processed_data_{VARIABLE}/"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'
HOURS_NAMES = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
               "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
POINT_LAT = -24.12
POINT_LON = -45.71
START_YEAR = "1980"
END_YEAR = "2022"

# Function to load time series data
def load_time_series(variable, lat_str, lon_str, start_year, end_year):
    """
    Loads the time series data for a given variable from a CSV file.

    This function searches for a specific CSV file in the processed data directory, reads it,
    and returns the time-indexed data.

    Args:
        variable (str): The variable name used to identify the relevant CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded time series data with a datetime index.
    """
    data_file = glob(os.path.join(PROCESSED_DATA_DIR, f"W_{variable}_{lat_str}_{lon_str}_{start_year}-{end_year}.csv"))[0]
    data = pd.read_csv(data_file, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

def create_lag_plots(data, variable, filename):
    """
    Creates and saves a lag plot for the given time series data.

    This function generates a lag plot that helps in visualizing autocorrelation in the data. The plot
    is saved as a PNG file.

    Args:
        data (pandas.DataFrame): The time series data to be plotted.
        variable (str): The name of the variable, used for labeling the plot and naming the output file.

    """
    plt.close("all")
    plt.figure(figsize=(9, 9))
    lag_plot(data, c=GRAY)
    plt.xlabel(f"${variable}_i$")  # Subscript for i, in math mode
    plt.ylabel(f"${variable}_{{i+1}}$")  # Subscript for i+1, in math mode
    plt.title(f"Lag 1 Plot of the Time Series ${variable}$")
    x_fit = np.linspace(data.min(), data.max(), len(data))
    y_fit = x_fit
    plt.plot(x_fit, y_fit, color=RED, label="Line $y = x$")
    plt.legend(loc="best")
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_{filename}.png"))

def create_lag_plots_with_lags(data, variable, lags, filename):
    """
    Creates and saves lag plots for specified lag values.

    This function generates multiple lag plots for the given time series data, each corresponding to a
    different lag value specified in the 'lags' list.

    Args:
        data (pandas.DataFrame): The time series data to be plotted.
        variable (str): The name of the variable, used for labeling the plots.
        lags (list of int): A list of integer lag values for which to create lag plots.

    """
    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(data, lag=lags[i], ax=ax, c=GRAY, alpha=0.8)
        ax.set_title(f"Lag {lags[i]}")
        ax.set_xlabel(f"${variable}_i$")
        ax.set_ylabel(f"${variable}_{{i+{lags[i]}}}$")
        x_fit = np.linspace(data.min(), data.max(), len(data))
        y_fit = x_fit
        ax.plot(x_fit, y_fit, color=RED, label="Line $y = x$")
    fig.suptitle(f"Lag Plots of the Time Series ${variable}$")
    fig.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_lags_{filename}.png"))

def plot_autocorrelation(data, variable, filename):
    """
    Plots and saves the autocorrelation function for the given time series data.

    This function calculates the autocorrelation function up to 192 lags and creates a plot
    showing these values, which is then saved as a PNG file.

    Args:
        data (pandas.DataFrame): The time series data for which to calculate autocorrelation.
        variable (str): The name of the variable, used for labeling the plot and naming the output file.

    """
    lag_acf = acf(data, nlags=192, fft=False)
    plt.figure(figsize=(12, 10))
    plt.plot(lag_acf, c=BLUE)
    plt.xlabel("Lag (hour)")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, linestyle="--", color=GRAY)
    plt.axhline(y=-1.96 / np.sqrt(len(data)), linestyle="--", color=GRAY)
    plt.axhline(y=1.96 / np.sqrt(len(data)), linestyle="--", color=GRAY)
    plt.title(f"Autocorrelation Function for the Time Series ${variable}$")
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_function_{filename}.png"))

def create_hourly_boxplots(data, variable, filename):
    """
    Creates and saves hourly boxplots for the given time series data.

    This function generates boxplots for each hour of the day, illustrating the distribution of data values
    at each hour. The resulting plot is saved as a PNG file.

    Args:
        data (pandas.DataFrame): The time series data to be visualized.
        variable (str): The name of the variable, used for labeling the plot and naming the output file.

    """
    plt.close("all")
    plt.figure(figsize=(16, 8))
    bp3 = plt.boxplot([data[data.index.hour == h]['normalized deviation']  for h in range(24)], labels=HOURS_NAMES, vert=True)
    plt.setp(bp3["medians"], color=RED, linewidth=3)
    plt.setp(bp3["boxes"], color=GRAY)
    plt.setp(bp3["whiskers"], color=GRAY, linestyle="--")
    plt.setp(bp3["fliers"], color=BLUE, marker="+")
    plt.xlabel("Hour")
    plt.ylabel("Normalized Deviations")
    plt.title(f"Hour-wise Boxplots for the Time Series ${variable}$")
    plt.savefig(os.path.join(FIGS_DIR, f"boxplot_hourwise_{filename}.png"))

def analyze_stationarity(data, variable, output_file):
    """
    Performs the Augmented Dickey-Fuller (ADF) test on the time series data to check for stationarity.

    The ADF test is used to determine the presence of unit root in the series, and hence, the stationarity.
    This is a critical step in the analysis as per the methodologies outlined in the research article 
    "The increase in intensity and frequency of surface air temperature extremes throughout the 
    western South Atlantic coast", which requires stationary time series for accurate analysis.

    Args:
        data (pandas.DataFrame): The time series data to be analyzed.
        variable (str): The name of the variable, used in the output for clarity.
        output_file (str): Path to the file where results will be saved.
    """

    with open(output_file, 'a') as file:
        file.write(f"Results of the Augmented Dickey-Fuller Test for {variable}\n")
        adf_test_result = adfuller(data, autolag="AIC")
        adf_output = pd.Series(adf_test_result[0:4], index=["Test Statistic:", "P-value:", "Number of Lags Used:", "Number of Observations Used:"])
        for key, value in adf_test_result[4].items():
            adf_output[f"Critical Value at {key} Level:"] = value
        file.write(f"{adf_output}\n")

        # Interpretation of ADF test result
        p_value = adf_test_result[1]
        if p_value <= 0.05:
            file.write(f"Conclusion: Reject the null hypothesis (H0). {variable} is stationary. Continue with the analysis.\n")
        else:
            file.write(f"Conclusion: Fail to reject the null hypothesis (H0). {variable} is non-stationary. Do not recommend continuing the analysis.\n")
            
def perform_ljung_box_test(data, variable, filename):
    """
    Performs the Ljung-Box test on the time series data to check for autocorrelation.

    This function assesses the degree of dependence in the data, which is vital for understanding the
    time series' characteristics in the context of the research article "The increase in intensity and 
    frequency of surface air temperature extremes throughout the western South Atlantic coast". The Ljung-Box 
    test checks whether there are significant autocorrelations in the data at different lags.

    Args:
        data (pandas.DataFrame): The time series data to be analyzed.
        variable (str): The name of the variable, used in the output for clarity.

    """
    lb_test_result = acorr_ljungbox(data, lags=192, boxpierce=False, return_df=True)
    print(lb_test_result)

    # Interpretation of Ljung-Box test result
    p_value = lb_test_result['lb_pvalue']
    plt.close("all")
    plt.figure(figsize=(8,8))
    plt.plot(p_value, color=GRAY)
    plt.axhline(y = 0.05, linestyle="--", color=RED)
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.title("Ljung-Box Test for $W$")
    plt.savefig(os.path.join(FIGS_DIR, f"lb_test_W_max_daily_{filename}.png"))

# Main function
def main():
    print(f"Performing stationarity analysis for {VARIABLE} at {POINT_LAT}, {POINT_LON}\n")
    lat, lon = np.round(float(POINT_LAT), 2), np.round(float(POINT_LON), 2)
    lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
    lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
    W = load_time_series(VARIABLE, lat_str, lon_str, START_YEAR, END_YEAR)
    
    filename = f"W_{VARIABLE.upper()}_{lat_str}_{lon_str}"
    create_lag_plots(W, VARIABLE, filename)
    print(f"lag plots for {VARIABLE} at {lat_str}, {lon_str} created")
    
    lag_values = [24, 48, 72, 96]
    create_lag_plots_with_lags(W, VARIABLE, lag_values, filename)
    print(f"{lag_values} lag plots for deseasonalized (W) {VARIABLE} at {lat}, {lon} created")
    
    plot_autocorrelation(W, VARIABLE, filename)
    print(f"autocorrelation plot for deseasonalized (W) {VARIABLE} at {lat_str}, {lon_str} created")
    
    create_hourly_boxplots(W, VARIABLE, filename)
    print(f"hourly boxplots for deseasonalized (W) {VARIABLE} at {lat}, {lon} created")
    
    output_file = os.path.join(PROCESSED_DATA_DIR, f"adf_stationarity_{filename}.txt")
    analyze_stationarity(W, VARIABLE, output_file)
    print(f"stationarity analysis results for deseasonalized (W) {VARIABLE} at {lat}, {lon} saved to {output_file}")
    
    perform_ljung_box_test(W, VARIABLE, filename)
    print(f"Ljung-Box test results for deseasonalized (W) {VARIABLE} at {lat}, {lon} saved to {output_file}")

if __name__ == "__main__":
    main()
