import xarray as xr
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FIGS_DIR = "./figures/"
os.makedirs(FIGS_DIR, exist_ok=True)

def configure_matplotlib():
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["black"])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

def load_data(file_pattern):
    ds = xr.open_mfdataset(file_pattern)
    ds['time'] = pd.to_datetime(ds.variables["time"])
    return ds

def plot_histogram(data, title, filename, xlabel, ylabel):
    plt.figure(figsize=(10, 10))
    plt.hist(np.ravel(data.compute()), bins=50, color='#023047')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axvline(data.mean(), color='red', label=f'Mean: {round(float(data.mean()), 2)}m')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def calculate_moving_stats(data, window_size):
    moving_avg = data.rolling(time=window_size, center=False).mean()
    moving_std = data.rolling(time=window_size, center=False).std()
    return moving_avg, moving_std

def plot_time_series(time, data, avg, std, title, filename):
    plt.figure(figsize=(16, 8))
    plt.plot(time, np.ravel(data.values), color="#023047", label="Original Series", lw=1)
    plt.plot(time, np.ravel(avg.values), color="red", label="Moving Average", lw=1.5)
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def main():
    configure_matplotlib()
    FIGS_DIR = "./figures/"
    os.makedirs(FIGS_DIR, exist_ok=True)

    # Load data
    ds = load_data("Hs*nc")
    swh = ds.variables["swh"]

    # Plot histogram
    histogram_title = f"{swh.attrs['long_name']} ({ds['time'][0].year} to {ds['time'][-1].year})"
    histogram_filename = os.path.join(FIGS_DIR, f"histogram_SWH_{ds['time'][0].year}-{ds['time'][-1].year}")
    plot_histogram(swh, histogram_title, histogram_filename, "", "Frequency")

    # Calculate and plot rolling statistics
    moving_avg, moving_std = calculate_moving_stats(swh, 8760)
    rolling_stats_title = f"SWH (m): {ds['time'][0].year} to {ds['time'][-1].year}"
    rolling_stats_filename = os.path.join(FIGS_DIR, f"rolling_stats_SWH_{ds['time'][0].year}-{ds['time'][-1].year}")
    plot_time_series(ds['time'], swh, moving_avg, moving_std, rolling_stats_title, rolling_stats_filename)


#######################

plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["black"])
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# Open all data into one dataset
ds = xr.open_mfdataset("Hs*nc")
time = ds.variables["time"] # variable time: from 1980 to 1989 in seconds since 1970-01-01
latitude = ds.variables["latitude"] # variable latitude: -24.0 degrees_north
longitude = ds.variables["longitude"] # variable longitude: -48.0 degrees_east
swh = ds.variables["swh"] # variable tas (air temperature near surface): Kelvin
datetime = pd.to_datetime(time)

plt.figure(figsize = (10, 10))
plt.hist(np.ravel(swh.compute()), bins = 50, color='#023047')
plt.xlabel("")
plt.title(f"{swh.attrs['long_name']}  ({datetime[0].year} to {datetime[-1].year})")
plt.ylabel("Frequency")
plt.axvline(swh.mean(), color = 'red', label = f'Mean: {round(float(swh.mean()), 2)}m')
plt.legend()
fname = f"histogram_SWH_{datetime[0].year}-{datetime[-1].year}"
plt.savefig(os.path.join(FIGS_DIR, fname))
print(f"Figure saved in {os.path.join(FIGS_DIR, fname)}")

# Define the periods
periods = {
    "1980-1989": slice("1980", "1989"),
    "1990-1999": slice("1990", "1999"),
    "2000-2009": slice("2000", "2009"),
    "2010-2019": slice("2010", "2019"),
    "2020-2022": slice("2020", "2022")
}

# determing rolling statistics
moving_avg = ds['swh'].rolling(time=8760, center=False).mean()
moving_std = ds['swh'].rolling(time=8760, center=False).std()

# plotting rolling statistics
plt.figure(figsize = (16, 8))
plt.plot(time, np.ravel(ds['swh'].values), color = "#023047", label = "Original Series", lw=1)
plt.plot(time, np.ravel(moving_avg.values), color = "red", label = "Moving Average", lw=1.5)
# plt.plot(time, np.ravel(moving_std.values), color = "black", label = "Moving Standard Deviation", lw=0.1)
plt.legend(loc = "best")
plt.title(f"SWH (m): {datetime[0].year} to {datetime[-1].year}")
fname = f"rolling_stats_SWH_{datetime[0].year}-{datetime[-1].year}"
plt.savefig(os.path.join(FIGS_DIR, fname))
print(f"Figure saved in {os.path.join(FIGS_DIR, fname)}")

# detrending the time series
Z = swh - moving_avg

# plotting the detrended time series
plt.figure(figsize = (16, 8))
plt.plot(time, np.ravel(Z.values), color="#023047")
plt.xlabel("")
plt.ylabel("Translated SWH (m)")
plt.title(f"Time Series Detrended by Subtraction of Its Moving Average (SWH {datetime[0].year} to {datetime[-1].year})")
plt.savefig(os.path.join(FIGS_DIR, f"detrended_SWH_{datetime[0].year}-{datetime[-1].year}.png"))

Z_squeezed = Z.squeeze(drop=True)  # This removes the longitude and latitude dimensions
Z_df = Z_squeezed.to_dataframe(name='Z')

# deseazonalizing the time series
indexes = Z_df.index
# W = pd.Series("NaN", index = indexes, dtype = float, name = "normalized deviation")
W = []
for m in range(1, 13):

    Z_month = Z_df[Z_df.index.month == m]
    Z_month_mean = Z_df[Z_df.index.month == m].mean()
    Z_month_std = Z_df[Z_df.index.month == m].std()

    W.append((Z_month - Z_month_mean) / Z_month_std)
W = pd.concat(W)
W = W.dropna()
W = W.rename(columns={"Z": "normalized deviation"})

# plotting the deseazonalized time series
plt.figure(figsize = (10, 8))
plt.plot(W.index, np.ravel(W.values), '-', color = "#023047", linewidth = 0.5)
plt.xlabel("Date")
plt.ylabel("Normalized Deviations")
plt.title(f"Time Series Deseazonalized SWH from {datetime[0].year} to {datetime[-1].year}")
plt.savefig(os.path.join(FIGS_DIR, f"deseazonalized_SWH_{datetime[0].year}-{datetime[-1].year}.png"))

# Plot histograms of W
plt.figure(figsize = (16, 8))
plt.hist(W, bins = 40, color="#023047")
plt.xlabel("Normalized Deviations")
plt.title("Histogram of W")
plt.axvline(float(W.mean()), color = "red", label = f"Mean = {round(float(W.mean().values), 2)}")
plt.legend()
plt.savefig(os.path.join(FIGS_DIR, f"histogram_W_{datetime[0].year}-{datetime[-1].year}.png"))

months_names = ["Jan", "Fev", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.figure(figsize = (16, 8))
bp2 = plt.boxplot([np.ravel(W[W.index.month == m].values) for m in range(1, 13)], labels = months_names, vert = True)
plt.setp(bp2["medians"], color = "red")
plt.setp(bp2["boxes"], color = "blue")
plt.setp(bp2["whiskers"], color = "blue", linestyle = "--")
plt.setp(bp2["fliers"], color = "black", marker = "+")
plt.xlabel("Month")
plt.ylabel("Normalized Deviations")
plt.title("Month-wise Boxplots for the Time Series $W$")
fname = f"boxplot_montlhy_W_{datetime[0].year}-{datetime[-1].year}.png"
plt.savefig(os.path.join(FIGS_DIR, fname))
print(f"Figure saved in {os.path.join(FIGS_DIR, fname)}")

W.to_csv(os.path.join(FIGS_DIR, f"W_{datetime[0].year}-{datetime[-1].year}.csv"))