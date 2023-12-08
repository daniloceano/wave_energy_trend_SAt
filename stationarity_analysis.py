import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import lag_plot

# Constants
FIGS_DIR = "./figures/"
PROCESSED_DATA_DIR = "./processed_data/"
VARIABLE = "swh"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'
HOURS_NAMES = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
               "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]

# Function to load time series data
def load_time_series(variable):
    data_file = glob(os.path.join(PROCESSED_DATA_DIR, "W_swh_24.12S_45.71W_1980-2022.csv"))[0]
    data = pd.read_csv(data_file, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

# Function to create lag plots
def create_lag_plots(data, variable):
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
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_{variable}.png"))

# Function to create lag plots for specified lags
def create_lag_plots_with_lags(data, variable, lags):
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
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_lags_{variable}.png"))

# Function to plot autocorrelation function
def plot_autocorrelation(data, variable):
    lag_acf = acf(data, nlags=192, fft=False)
    plt.figure(figsize=(12, 10))
    plt.plot(lag_acf, c=BLUE)
    plt.xlabel("Lag (hour)")
    plt.ylabel("Autocorrelation")
    plt.axhline(y=0, linestyle="--", color=GRAY)
    plt.axhline(y=-1.96 / np.sqrt(len(data)), linestyle="--", color=GRAY)
    plt.axhline(y=1.96 / np.sqrt(len(data)), linestyle="--", color=GRAY)
    plt.title(f"Autocorrelation Function for the Time Series ${variable}$")
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_function_{variable}.png"))

# Function to create hour-wise boxplots
def create_hourly_boxplots(data, variable):
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
    plt.savefig(os.path.join(FIGS_DIR, f"boxplot_hourwise_{variable}.png"))

# Function to analyze stationarity using ADF test
# Function to analyze stationarity using ADF test and print results
def analyze_stationarity(data, variable):
    print(f"Results of the Augmented Dickey-Fuller Test for {variable}\n")
    adf_test_result = adfuller(data, autolag="AIC")
    adf_output = pd.Series(adf_test_result[0:4], index=["Test Statistic:", "P-value:", "Number of Lags Used:", "Number of Observations Used:"])
    for key, value in adf_test_result[4].items():
        adf_output[f"Critical Value at {key} Level:"] = value
    print(adf_output)

    # Interpretation of ADF test result
    p_value = adf_test_result[1]
    if p_value <= 0.05:
        print(f"Conclusion: Reject the null hypothesis (H0). {variable} is stationary. Continue with the analysis.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis (H0). {variable} is non-stationary. Do not recommend continuing the analysis.")

# Function to perform Ljung-Box Test and print results
def perform_ljung_box_test(data, variable):
    print(f"Results of the Ljung-Box Test for {variable}\n")
    lb_test_result = acorr_ljungbox(data, lags=192, boxpierce=False, return_df=True)
    print(lb_test_result)

    # Interpretation of Ljung-Box test result
    p_value = lb_test_result['lb_pvalue']
    if (p_value >= 0.05).any():
        print(f"Conclusion: Fail to reject the null hypothesis (H0). {variable} is independently distributed. Do not recommend continuing the analysis.")        
    else:
        print(f"Conclusion: Reject the null hypothesis (H0). {variable} exhibits serial correlation. Continue with the analysis.")

# Main function
def main():
    W = load_time_series(VARIABLE)
    
    create_lag_plots(W, VARIABLE)
    
    lag_values = [24, 48, 72, 96]
    create_lag_plots_with_lags(W, VARIABLE, lag_values)
    
    plot_autocorrelation(W, VARIABLE)
    
    create_hourly_boxplots(W, VARIABLE)
    
    analyze_stationarity(W, VARIABLE)
    
    perform_ljung_box_test(W, VARIABLE)

if __name__ == "__main__":
    main()
