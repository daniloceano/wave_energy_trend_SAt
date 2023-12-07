import os 
import pandas as pd
import numpy as np
from glob import glob
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt

# Configuration
FIGS_DIR = "./figures/"
PROCESSED_DATA_DIR = "./processed_data/"
VARIABLE = "swh"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'

W_file = glob(os.path.join(PROCESSED_DATA_DIR, f"W_{VARIABLE}*.csv"))[0]
W = pd.read_csv(W_file, index_col=0)

# Analyzing Stationarity of the Time Series 𝑊 by Means of the Augmented Dickey-Fuller (ADF) Test
print("Results of the Augmented Dickey-Fuller Test\n")
adf_test = adfuller(W, autolag = "AIC") # AIC is the "Akaike Information Criterion"
adf_output = pd.Series(adf_test[0:4], index = ["test statistic:", "p-value:", "number of lags used:", "number of observations used:"])
for key, value in adf_test[4].items():
    adf_output["critical value for the test statistic at the {} level:".format(key)] = value
print(adf_output)

# Checking Autocorrelation for the Time Series  𝑊
plt.figure(figsize = (9, 9))
lag_plot(W)
plt.xlabel("W_i")
plt.ylabel("W_i+1")
plt.title("Lag 1 Plot of the Time Series $W$")
x_fit = np.linspace(-5, 5, 518)
y_fit = x_fit
plt.plot(x_fit, y_fit, color=RED, label="Line $y = x$")
plt.legend(loc="best")
plt.show()
 