import pandas as pd
import pymannkendall as mk
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import os
import sys
from glob import glob 

# Constants
VARIABLE = "swh"
FIGS_DIR = f"./figures/E_{VARIABLE}/"
PROCESSED_DATA_DIR = f"./processed_data_{VARIABLE}/"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'
ALPHA = 0.05
POINT_LAT = -24.12
POINT_LON = -45.71
START_YEAR = "1980"
END_YEAR = "2022"

os.makedirs(FIGS_DIR, exist_ok=True)

def load_E_time_series(variable, lat_str, lon_str, start_year, end_year):
    """
    Loads the time series data for a given variable from a CSV file.

    This function searches for a specific CSV file in the processed data directory, reads it,
    and returns the time-indexed data.

    Args:
        variable (str): The variable name used to identify the relevant CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded time series data with a datetime index.
    """
    data_file = glob(os.path.join(PROCESSED_DATA_DIR, f"E_{variable}_{lat_str}_{lon_str}_{start_year}-{end_year}.csv"))[0]
    data = pd.read_csv(data_file, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

def read_u0_from_file(variable, lat_str, lon_str, start_year, end_year):
    """
    Reads the threshold value u0 from a text file.

    Args:
        variable (str): The variable name used to identify the relevant text file.
        lat_str (str): Latitude string for file identification.
        lon_str (str): Longitude string for file identification.
        start_year (str): Start year for file identification.
        end_year (str): End year for file identification.

    Returns:
        float: The value of u0 read from the file.
    """
    u0_filename = os.path.join(PROCESSED_DATA_DIR, f"u0_{variable}_{lat_str}_{lon_str}_{start_year}-{end_year}.txt")
    try:
        with open(u0_filename, 'r') as file:
            line = file.readline()
            u0_value = float(line.split(":")[1].strip())
        return u0_value
    except FileNotFoundError:
        print(f"File not found: {u0_filename}")
        return None
    except ValueError:
        print("Error in reading the u0 value from the file.")
        return None

# Import excess data
lat, lon = np.round(float(POINT_LAT), 2), np.round(float(POINT_LON), 2)
lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
E = load_E_time_series(VARIABLE, lat_str, lon_str, START_YEAR, END_YEAR)
filename = f"E_{VARIABLE}_{lat_str}_{lon_str}"

# Read u0 from file
u0 = read_u0_from_file(VARIABLE, lat_str, lon_str, START_YEAR, END_YEAR)
if u0 is not None:
    print(f"Read threshold u0: {u0}")
else:
    print("Failed to read threshold u0 from file.")
    sys.exit(1)

# plotting the normalized histogram for exceedances over the threshold series and the GPD probability density function
c, loc, scale = genpareto.fit(E - u0)  # Fit the GPD to the exceedances data
xsi, sigma = c, scale # GPD parameters
min_excess, max_excess = E.min(), E.max() # Minimum and maximum exceedances
plt.figure(figsize=(16, 8))
plt.hist(E, bins=40, density=True, color=GRAY)
plt.xlabel("Excesses")
plt.title("Histogram for Exceedances Over the Threshold Series $E$ and The Adjusted GPD Probability Density Function")
x_fit = np.linspace(min_excess, max_excess, 500)  # Adjust the number of points as needed
h_fit = (1/sigma) * (1 + xsi * ((x_fit)/sigma))**(-1-1/xsi)  # h(x_fit) = h_fit
plt.plot(x_fit , h_fit, color="red", label="GPD Probability Density Function")
plt.legend(loc="best")
plt.savefig(os.path.join(FIGS_DIR, f"histogram_E_{filename}.png"))
print(f"Histogram for exceedances (E) {VARIABLE} at {lat}, {lon} created")

# Apply Mann-Kendall Test to check if the intensities have some linear trend.
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(E.values, alpha=0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index=["trend:", "p-value:"])
print(mk_output)
output_file = os.path.join(PROCESSED_DATA_DIR, f"kendall-test_{filename}.txt")
with open(output_file, 'w') as file:
    file.write(mk_output.to_string())
print(f"Mann-Kendall test results saved to {output_file}")

# Plotting the exceedances over the threshold series and the linear trend function
plt.close("all")
plt.figure(figsize=(16, 8))
st1 = plt.stem(E.index, E.values)
plt.setp(st1[0], color=RED, marker="o", markersize=2)
plt.setp(st1[1], color=GRAY, linestyle="-")
plt.setp(st1[2], visible=False)
plt.xlabel("Date")
plt.ylabel("Excesses")
plt.title("Exceedances Over the Threshold Series $E$ and the Linear Trend Function")
x = (E.index - E.index[0]).days.values.reshape(-1, 1)
y = E.values
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(E.index, y_fit, color=RED, label="Line $y = $" + str(float(model.coef_)) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.savefig(os.path.join(FIGS_DIR, f"trend_E_{filename}.png"))
print(f"Trend for exceedances (E) {VARIABLE} at {lat}, {lon} created")