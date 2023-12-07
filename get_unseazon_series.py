import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
FIGS_DIR = "./figures/"
PROCESSED_DATA_DIR = "./processed_data/"
VARIABLE = "swh"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'

# Ensure directories exist
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def configure_matplotlib():
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[GRAY])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

def load_data(file_pattern):
    ds = xr.open_mfdataset(file_pattern)
    ds['time'] = pd.to_datetime(ds.variables["time"])
    return ds

def plot_and_save(data, plot_type, title, xlabel, ylabel, filename, time=None, avg=None, std=None, **kwargs):
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
    
    filename_path = os.path.join(FIGS_DIR, f"{plot_type}_{filename}.png")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.savefig(filename_path)
    plt.close()

def calculate_moving_stats(data, window_size):
    return data.rolling(center=False, window=window_size).mean(), data.rolling(center=False, window=window_size).std()

def deseasonalize_data(data, index):
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
    X = ds.squeeze(['longitude', 'latitude']).to_dataframe().drop(columns=["latitude", "longitude"])
    datetime = pd.to_datetime(X.index)

    # Parameters
    variable_long_name = ds.variables[VARIABLE].attrs["long_name"]
    plot_title = f"{variable_long_name} ({datetime[0].year} to {datetime[-1].year})"
    filename = VARIABLE.upper()

    # Compute values for main variable and plot
    data = X[VARIABLE]
    plot_and_save(data, 'histogram', plot_title, VARIABLE.upper(), "Frequency", filename)
    plot_and_save(data, 'boxplot', plot_title, "month", VARIABLE.upper(), filename)
    plot_and_save(data, 'time_series', plot_title, "", VARIABLE.upper(), filename, time=datetime)

    # Calculate and plot rolling statistics
    moving_avg, moving_std = calculate_moving_stats(X[VARIABLE], 8760)
    filename_rolling = f"rolling_{VARIABLE.upper()}"
    plot_and_save(X[VARIABLE], 'time_series', plot_title, "",
                f"{VARIABLE.upper()} (m)", filename_rolling, 
                  time=datetime, avg=moving_avg, std=moving_std)

    # Detrending data
    Z = X[VARIABLE] - moving_avg
    Z = Z.dropna()

    # Plot detrended data
    detrended_filename = f"detrended_{VARIABLE.upper()}"
    deseasonalized_title = f"Time Series Detrended by Subtraction of Its Moving Average ($Z$) \n{plot_title}"
    plot_and_save(Z, 'time_series', deseasonalized_title, "Date", f"Translated {VARIABLE.upper()}",
                  detrended_filename, time=Z.index)

    # Deseazonalizing the time series
    W = deseasonalize_data(Z, X.index)
    W.rename('normalized deviation', inplace=True)

    # Plot deseazonalized time series
    deseasonalized_filename = f"deseasonalized_{VARIABLE.upper()}"
    deseasonalized_title = f"Deseazonalized ($W$) \n{plot_title}"
    plot_and_save(W, 'histogram', deseasonalized_title, "Normalized Deviations", "Frequency", deseasonalized_filename)
    plot_and_save(W, 'boxplot', deseasonalized_title, "Month", "Normalized Deviations", deseasonalized_filename)
    plot_and_save(W, 'time_series', deseasonalized_title, "Date", "Normalized Deviations",
                  deseasonalized_filename, time=W.index)

    # Save processed data
    W.to_csv(os.path.join(PROCESSED_DATA_DIR, f"W_{VARIABLE}_{datetime[0].year}-{datetime[-1].year}.csv")
    print(f"Processed data saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
