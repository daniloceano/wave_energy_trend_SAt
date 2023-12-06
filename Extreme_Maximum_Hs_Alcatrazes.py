import xarray as xr
import pandas as pd
import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

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

plt.figure(figsize = (16, 8))
plt.hist(np.ravel(swh.compute()), bins = 40, color='#023047')
plt.xlabel("Temperature (ºC)")
plt.title("Histogram (SAT in Iguape from 1980 to 2019)")
plt.ylabel("Frequency")
plt.show()


# Define the periods
periods = {
    "1980-1989": slice("1980", "1989"),
    "1990-1999": slice("1990", "1999"),
    "2000-2009": slice("2000", "2009"),
    "2010-2019": slice("2010", "2019"),
    "2020-2022": slice("2020", "2022")
}

# Create subplots
fig, axes = plt.subplots(nrows=len(periods), ncols=1, figsize=(12, 20))
months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for i, (period, time_range) in enumerate(periods.items()):
    # Select data for the period
    period_data = ds.sel(time=time_range)['swh'].to_dataframe().reset_index()
    period_data.set_index('time', inplace=True)

    # Create boxplot for each month in the period
    axes[i].boxplot([period_data[period_data.index.month == m]['swh'] for m in range(1, 13)], labels=months_names, vert=True)

    # Customizing boxplot appearance
    axes[i].set_title(f"SWH from {period}")
    axes[i].set_xlabel("Month")
    axes[i].set_ylabel("SWH (Value)")  # Replace "Value" with the appropriate unit/measurement

plt.tight_layout()
plt.show()


# determing rolling statistics
moving_avg = ds['swh'].rolling(time=8760, center=False).mean()
moving_std = ds['swh'].rolling(time=8760, center=False).std()

# plotting rolling statistics
plt.figure(figsize = (16, 8))
plt.plot(time, np.ravel(ds['swh'].values), label = "Original Series")
plt.plot(moving_avg, color = "red", label = "Moving Average")
plt.plot(moving_std, color = "black", label = "Moving Standard Deviation")
plt.xlabel("Date")
plt.ylabel("Temperature (ºC)")
plt.legend(loc = "best")
plt.title("Original Series, Moving Average, and Moving Standard Deviation (Air Temperature in Iguape from 1980 to 2019)")
plt.show()

# detrending the time series
swh_detrend = swh - moving_avg

# plotting the detrended time series
plt.figure(figsize = (16, 8))
plt.plot(time, np.ravel(swh_detrend.values))
plt.xlabel("Date")
plt.ylabel("Translated Temperature (ºC)")
plt.title("Time Series Detrended by Subtraction of Its Moving Average (Air Temperature in Iguape from 1980 to 2019)")
plt.show()

swh_detrend_squeezed = swh_detrend.squeeze(drop=True)  # This removes the longitude and latitude dimensions
swh_detrend_df = swh_detrend_squeezed.to_dataframe(name='swh_detrend')

# deseazonalizing the time series
indexes = swh_detrend_df.index
# W = pd.Series("NaN", index = indexes, dtype = float, name = "normalized deviation")
W = []
for m in range(1, 13):

    swh_detrend_month = swh_detrend_df[swh_detrend_df.index.month == m]
    swh_detrend_month_mean = swh_detrend_df[swh_detrend_df.index.month == m].mean()
    swh_detrend_month_std = swh_detrend_df[swh_detrend_df.index.month == m].std()

    W.append((swh_detrend_month - swh_detrend_month_mean) / swh_detrend_month_std)
W = pd.concat(W)
W = W.dropna()
W = W.rename(columns={"swh_detrend": "normalized deviation"})

# plotting the deseazonalized time series
plt.figure(figsize = (16, 8))
plt.plot(W)
plt.xlabel("Date")
plt.ylabel("Normalized Deviations")
plt.title("Time Series Deseazonalized (Air Temperature in Iguape from 1980 to 2019)")
plt.show()

plt.figure(figsize = (16, 8))
plt.hist(W, bins = 40)
plt.xlabel("Normalized Deviations")
plt.title("Histogram of the Time Series $W$")
plt.show()

plt.figure(figsize = (16, 8))
bp2 = plt.boxplot([np.ravel(W[W.index.month == m].values) for m in range(1, 13)], labels = months_names, vert = True)
plt.setp(bp2["medians"], color = "red")
plt.setp(bp2["boxes"], color = "blue")
plt.setp(bp2["whiskers"], color = "blue", linestyle = "--")
plt.setp(bp2["fliers"], color = "black", marker = "+")
plt.xlabel("Month")
plt.ylabel("Normalized Deviations")
plt.title("Month-wise Boxplots for the Time Series $W$")
plt.show()

# performing Augmented Dickey-Fuller (ADF) Test
print("Results of the Augmented Dickey-Fuller Test\n")
adf_test = adfuller(W, autolag = "AIC") # AIC is the "Akaike Information Criterion"
adf_output = pd.Series(adf_test[0:4], index = ["test statistic:", "p-value:", "number of lags used:", "number of observations used:"])
for key, value in adf_test[4].items():
    adf_output["critical value for the test statistic at the {} level:".format(key)] = value
print(adf_output)

plt.figure(figsize = (9, 9))
lag_plot(W)
plt.xlabel("W_i")
plt.ylabel("W_i+1")
plt.title("Lag 1 Plot of the Time Series $W$")
x_fit = np.linspace(-5, 5, 518)
y_fit = x_fit
plt.plot(x_fit, y_fit, color = "red", label = "Line $y = x$")
plt.legend(loc = "best")
plt.show()

fig, axes = plt.subplots(2, 2, figsize = (16, 16), sharex = True, sharey = True, dpi = 100)
for l, ax in enumerate(axes.flatten()[:4]):
    lag_plot(W, lag = 24*(l+1), ax = ax)
    ax.set_title("Lag " + str(24*(l+1)))
    ax.set_xlabel("W_i")
    ax.set_ylabel("W_i+" + str(24*(l+1)))
fig.suptitle("Lag Plots of the Time Series $W$")
plt.show()

lag_acf = acf(W, nlags = 192, fft = False)
plt.figure(figsize = (16, 8))
plt.plot(lag_acf)
plt.xlabel("Lag (hour)")
plt.ylabel("Autocorrelation")
plt.axhline(y = 0, linestyle = "--", color = "gray")
plt.axhline(y = -1.96/np.sqrt(len(W)), linestyle = "--", color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(W)), linestyle = "--", color = "gray")
plt.title("Autocorrelation Function for the Time Series $W$")
plt.show()

hours_names = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
plt.figure(figsize = (16, 8))
bp3 = plt.boxplot([W[W.index.hour == h] for h in range(0, 24)], labels = hours_names, vert = True)
plt.setp(bp3["medians"], color = "red")
plt.setp(bp3["boxes"], color = "blue")
plt.setp(bp3["whiskers"], color = "blue", linestyle = "--")
plt.setp(bp3["fliers"], color = "black", marker = "+")
plt.xlabel("Hour")
plt.ylabel("Normalized Deviations")
plt.title("Hour-wise Boxplots for the Time Series $W$")
plt.show()

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(W, lags = 192, boxpierce = False, return_df = True)
print(lb_test)

# installing POT R package (package that the software uses by means of rpy2 to compute GPD estimates)
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

base = importr("base")
utils = importr("utils")
utils.chooseCRANmirror(ind = 1)
utils.install_packages("POT") #installing POT package

# importing the "thresholdmodeling" package
from thresholdmodeling import thresh_modeling

plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["black"])
plt.rcParams.update({'font.size': 12})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# constructing the maximum daily normalized deviation time series from the time series W
days = int(len(W[1:])/24) # number of days from 1980-12-31 to 2019-12-31
W_max_daily = pd.Series() # creating an empty time series
for d in range(0, days):
    W_max_daily[W[(1+24*d):(25+24*d)].idxmax()] = W[(1+24*d):(25+24*d)].max()
W_max_daily = W_max_daily.rename("maximum daily normalized deviation")

# Observação: Nós poderíamos ter construído a série do desvio máximo diário com o método .resample:
# W_max_daily = W.resample("24H").max()
# Entretanto, o índice seria modificado: disposto dia-a-dia, sem o registro da hora. Isso não é bom para nosso problema, pois
# após toda a análise a ser feita, queremos ser capazes de voltar à série temporal original X e ver quais são os valores de
# temperatura dos extremos.

lag_acf = acf(W_max_daily, nlags = 192, fft = False)
plt.figure(figsize = (16, 8))
plt.plot(lag_acf)
plt.xlabel("Lag (day)")
plt.ylabel("Autocorrelation")
plt.axhline(y = 0, linestyle = "--", color = "gray")
plt.axhline(y = -1.96/np.sqrt(len(W_max_daily)), linestyle = "--", color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(W_max_daily)), linestyle = "--", color = "gray")
plt.title("Autocorrelation Function for the Time Series $W$_max_daily")
plt.show()

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(W_max_daily, lags = 192, boxpierce = False, return_df = True)
print(lb_test)

# plotting the mean excess (mean residual life) function with a confidence level of 5%
plt.figure(figsize = (16, 8))
plt.axvline(W_max_daily.quantile(0.97), color = "red", linestyle = "--", label = "Threshold Line $u_{0}$ = 0.97-quantile = " + str(W_max_daily.quantile(0.97)))
plt.legend(loc = "best")
thresh_modeling.MRL(W_max_daily, 0.05)
plt.show()

# plotting the shape and modified scale parameters stability with a confidence level of 5%
thresh_modeling.Parameter_Stability_plot(W_max_daily, 0.05)

# chosen threshold
u0 = W_max_daily.quantile(0.97)

# plotting the time series W_max_daily and the threshold line
plt.figure(figsize = (16, 8))
plt.plot(W_max_daily, label = "Time Series $W$_max_daily")
plt.axhline(u0, color = "red", linestyle = "--", label = "Threshold Line $u_{0}$ = 0.97-quantile = " + str(u0))
plt.xlabel("Date")
plt.ylabel("Normalized Deviations")
plt.legend(loc = "best")
plt.title("Time Series $W$_max_daily and the Threshold Line")
plt.show()

# exceedances over the threshold series
E = W_max_daily[W_max_daily.values > u0] - u0

lag_acf = acf(E, nlags = 192, fft = False)
plt.figure(figsize = (16, 8))
plt.plot(lag_acf)
plt.xlabel("Lag (day)")
plt.ylabel("Autocorrelation")
plt.axhline(y = 0, linestyle = "--", color = "gray")
plt.axhline(y = -1.96/np.sqrt(len(E)), linestyle = "--", color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(E)), linestyle = "--", color = "gray")
plt.title("Autocorrelation Function for Exceedances Over the Threshold Series $E$")
plt.show()

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(E, lags = 192, boxpierce = False, return_df = True)
print(lb_test)

# fitting the time series W_max_daily data to a GPD model with the chosen threshold u0
thresh_modeling.gpdfit(W_max_daily, u0, "mle")

# model checking (empirical versus model): plotting quantile-quantile, probability-probability and cumulative-cumulative 
# functions with the chosen threshold u0 and with a confidence level of 5%
thresh_modeling.qqplot(W_max_daily, u0, "mle", 0.05)
thresh_modeling.ppplot(W_max_daily, u0, "mle", 0.05)
thresh_modeling.gpdcdf(W_max_daily, u0, "mle", 0.05)

import pymannkendall as mk

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# GPD parameters
xsi = -0.05767 # shape parameter
sigma = 0.68316 # scale parameter

# plotting the normalized histogram for exceedances over the threshold series and the GPD probability density function
plt.figure(figsize = (16, 8))
plt.hist(E, bins = 40, density = True)
plt.xlabel("Excesses")
plt.title("Histogram for Exceedances Over the Threshold Series $E$ and The Adjusted GPD Probability Density Function")
x_fit = np.linspace(2.918, 5, 518)
h_fit = (1/sigma) * (1 + xsi * ((x_fit - u0)/sigma))**(-1-1/xsi) # h(x_fit) = h_fit
plt.plot(x_fit - u0, h_fit, "r", label = "GPD Probability Density Function")
plt.legend(loc = "best")
plt.show

# performing Mann-Kendall Trend (MK) Test
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(E.values, alpha = 0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index = ["trend:", "p-value:"])
print(mk_output)

# plotting the exceedances over the threshold series and the linear trend function
plt.figure(figsize = (16, 8))
st1 = plt.stem(E.index, E.values, use_line_collection = True)
plt.setp(st1[0], color = "orange", marker = "o", markersize = 2)
plt.setp(st1[1], color = "black", linestyle = "-")
plt.setp(st1[2], visible = False)
plt.xlabel("Date")
plt.ylabel("Excesses")
plt.title("Exceedances Over the Threshold Series $E$ and the Linear Trend Function")
x = ((E.index - E.index[0]).astype("timedelta64[h]")/24).values.reshape(-1, 1)
y = E.values
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(E.index, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.show()

# calculating the time elapsed between consecutive extreme excesses
differences_days = []
total_number_extreme_days = len(E)
for k in range(0, total_number_extreme_days-1):
    differences_days.append(E.index[k+1] - E.index[k])
time_elapsed_days = pd.Series(differences_days)
time_elapsed_days = time_elapsed_days.rename("time elapsed between consecutive extreme excesses")

# plotting the normalized histogram for the time elapsed between consecutive extreme excesses,
# fitting the time elapsed between consecutive extreme excesses data to an exponential probability density function and
# plotting the exponential probability density function
plt.figure(figsize = (16, 8))
plt.hist(time_elapsed_days.astype("timedelta64[h]")/24, bins = 40, density = True)
plt.xlabel("Time Elapsed (day)")
plt.title("Histogram for the Time Elapsed Between Consecutive Extreme Excesses and The Adjusted Exponential Probability Density Function")

from scipy.optimize import curve_fit
t_min = (time_elapsed_days.astype("timedelta64[h]").min())/24
t_max = (time_elapsed_days.astype("timedelta64[h]").max())/24
h = 1/24
I = np.arange(t_min, t_max+h, h, dtype = float)
hist, bin_edges = np.histogram(time_elapsed_days.astype("timedelta64[h]")/24, bins = I, density = True)
x = np.arange(t_min, t_max, h, dtype = float)
y = hist

def f(t, c0):
    return c0 * np.exp(-c0 * t)

print("Results of the Fit\n")
lamb, var = curve_fit(f, x, y)
fit_output = pd.Series([lamb, var], index = ["lambda:", "variance:"])
print(fit_output)

t_fit = np.linspace(2.5, t_max, 518)
f_fit = f(t_fit, lamb) # f(t_fit) = f_fit
plt.plot(t_fit, f_fit, "r", label = "Exponential Probability Density Function")
plt.legend(loc = "best")
plt.show()

lag_acf = acf(time_elapsed_days, nlags = 192, fft = False)
plt.figure(figsize = (16, 8))
plt.plot(lag_acf)
plt.xlabel("Lag (day)")
plt.ylabel("Autocorrelation")
plt.axhline(y = 0, linestyle = "--", color = "gray")
plt.axhline(y = -1.96/np.sqrt(len(time_elapsed_days)), linestyle = "--", color = "gray")
plt.axhline(y = 1.96/np.sqrt(len(time_elapsed_days)), linestyle = "--", color = "gray")
plt.title("Autocorrelation Function for the Time Elapsed Between Consecutive Extreme Excesses")
plt.show()

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(time_elapsed_days, lags = 192, boxpierce = False, return_df = True)
print(lb_test)

# performing Mann-Kendall Trend (MK) Test
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(time_elapsed_days.astype("timedelta64[h]")/24, alpha = 0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index = ["trend:", "p-value:"])
print(mk_output)

# plotting the time elapsed between consecutive extreme excesses and the linear trend function
plt.figure(figsize = (16, 8))
st2 = plt.stem(time_elapsed_days.index, time_elapsed_days.astype("timedelta64[h]").values/24, use_line_collection = True)
plt.setp(st2[0], color = "orange", marker = "o", markersize = 2)
plt.setp(st2[1], color = "black", linestyle = "-")
plt.setp(st2[2], visible = False)
plt.xlabel("Gap Number")
plt.ylabel("Time Elapsed (day)")
plt.title("Time Elapsed Between Consecutive Extreme Excesses and the Linear Trend Function")
x = (time_elapsed_days.index - time_elapsed_days.index[0]).values.reshape(-1, 1)
y = (time_elapsed_days.astype("timedelta64[h]").values)/24
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(time_elapsed_days.index, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.show()

# parameters
block_size = 365 # number of days in the block/number of observations per year
N = 5 # number of years of interest
return_period = N * block_size # N–year return period

# N–year return level for the time series W_max_daily with a confidence level of 5%
thresh_modeling.return_value(W_max_daily, u0, 0.05, block_size, return_period, "mle")

i = E.index
X_e = pd.Series("NaN", index = i, dtype = float, name = "extreme maximum temperatures")
for i in X_e.index:
    X_e[i] = X[i]

print("From 1980-12-31 to 2019-12-31 there are {} extreme maximum temperatures:\n".format(len(X_e)))
for i in X_e.index:
    print("Height of {:.2f}m in {}.".format(X_e[i], i))

plt.figure(figsize = (16, 8))
bp4 = plt.boxplot([X_e[X_e.index.month == m] for m in range(1, 13)], labels = months_names, vert = True)
plt.setp(bp4["medians"], color = "red")
plt.setp(bp4["boxes"], color = "blue")
plt.setp(bp4["whiskers"], color = "blue", linestyle = "--")
plt.setp(bp4["fliers"], color = "black", marker = "+")
plt.xlabel("Month")
plt.ylabel("Temperature (ºC)")
plt.title("Month-wise Boxplots for the Extreme Maximum Temperatures Series from 1980-12-31 to 2019-12-31")
plt.show()

plt.figure(figsize = (16, 8))
bp5 = plt.boxplot([E[E.index.month == m] for m in range(1, 13)], labels = months_names, vert = True)
plt.setp(bp5["medians"], color = "red")
plt.setp(bp5["boxes"], color = "blue")
plt.setp(bp5["whiskers"], color = "blue", linestyle = "--")
plt.setp(bp5["fliers"], color = "black", marker = "+")
plt.xlabel("Month")
plt.ylabel("Excesses")
plt.title("Month-wise Boxplots for the Exceedances Over the Threshold Series")
plt.show()

years = np.arange(1980, 2020, dtype = int)
plt.figure(figsize = (16, 8))
bp6 = plt.boxplot([E[E.index.year == y] for y in years], labels = years, vert = True)
plt.xticks(rotation = 90)
plt.setp(bp6["medians"], color = "red")
plt.setp(bp6["boxes"], color = "blue")
plt.setp(bp6["whiskers"], color = "blue", linestyle = "--")
plt.setp(bp6["fliers"], color = "black", marker = "+")
plt.xlabel("Year")
plt.ylabel("Excesses")
plt.title("Year-wise Boxplots for the Exceedances Over the Threshold Series")
plt.show()

# counting by month
months = np.arange(1, 13, dtype = int)
countd_months = np.zeros(12, dtype = int)
for m in months:
    countd_months[m-1] = E[E.index.month == m].count()

# creating barplot
plt.figure(figsize = (16, 8))
barWidth = 0.45
plt.bar(months, countd_months, width = barWidth)
plt.xticks(months, months_names)
for m in months:
    plt.text(x = months[m-1]-0.08, y = countd_months[m-1]+0.5, s = countd_months[m-1], size = 12)
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Number of Extreme Maximum Daily Temperatures by Month from 1980-12-31 to 2019-12-31")
plt.show()

# counting by season
summerd = ((E.index.month == 12) & (E.index.day >= 21)) + (E.index.month == 1) + (E.index.month == 2) + ((E.index.month == 3) & (E.index.day <= 20))
autumnd = ((E.index.month == 3) & (E.index.day >= 21)) + (E.index.month == 4) + (E.index.month == 5) + ((E.index.month == 6) & (E.index.day <= 20))
winterd = ((E.index.month == 6) & (E.index.day >= 21)) + (E.index.month == 7) + (E.index.month == 8) + ((E.index.month == 9) & (E.index.day <= 20))
springd = ((E.index.month == 9) & (E.index.day >= 21)) + (E.index.month == 10) + (E.index.month == 11) + ((E.index.month == 12) & (E.index.day <= 20))
seasons = np.arange(0, 4, dtype = int)
countd_seasons = np.zeros(4, dtype = int)
countd_seasons[0] = E[summerd].count()
countd_seasons[1] = E[autumnd].count()
countd_seasons[2] = E[winterd].count()
countd_seasons[3] = E[springd].count()

# creating barplot
seasons_names = ["Summer", "Autumn", "Winter", "Spring"]
plt.figure(figsize = (16, 8))
barWidth = 0.45
plt.bar(seasons, countd_seasons, width = barWidth)
plt.xticks(seasons, seasons_names)
for s in seasons:
    plt.text(x = seasons[s]-0.025, y = countd_seasons[s]+0.5, s = countd_seasons[s], size = 12)
plt.xlabel("Season")
plt.ylabel("Count")
plt.title("Number of Extreme Maximum Daily Temperatures by Season from 1980-12-31 to 2019-12-31")
plt.show()

# counting by year
countd_years = np.zeros(40, dtype = int)
for y in years:
    countd_years[y-1980] = E[E.index.year == y].count()
total_number_extreme_days = sum(countd_years) # = len(E)
print("From 1980-12-31 to 2019-12-31 there are {} extreme maximum daily temperatures.".format(total_number_extreme_days))

print("\n")

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(countd_years, lags = 3, boxpierce = False, return_df = True)
print(lb_test)

print("\n")

# performing Mann-Kendall Trend (MK) Test
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(countd_years, alpha = 0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index = ["trend:", "p-value:"])
print(mk_output)

print("\n")

# creating barplot
plt.figure(figsize = (16, 8))
barWidth = 0.45
plt.bar(years, countd_years, width = barWidth)
plt.xticks(years, years, rotation = 90)
for y in years:
    plt.text(x = years[y-1980]-0.3, y = countd_years[y-1980]+0.5, s = countd_years[y-1980], size = 12)
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Number of Extreme Maximum Daily Temperatures by Year from 1980-12-31 to 2019-12-31 and the Linear Trend Function")
x = (years - years[0]).reshape(-1, 1)
y = countd_years
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(years, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.show()

# accumulated number of extreme days
accumulated_countd_years = np.zeros(40, dtype = int)
for j in range(0, 40):
    accumulated_countd_years[j] = sum(countd_years[k] for k in range(0, j+1))

# plotting the accumulated number of extreme days
plt.figure(figsize = (16, 8))
plt.plot(years, accumulated_countd_years, linestyle = ":", marker = "o")
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Accumulated Number of Extreme Maximum Daily Temperatures by Year from 1980-12-31 to 2019-12-31")
plt.show()

# about events: identification, initial date, final date and duration of each event
days = len(W_max_daily.index) # number of days from 1980-12-31 to 2019-12-31
n = 0
events_durations = []
events_id = []
events_fd = []
for d in range(0, days):
    if W_max_daily[W_max_daily.index[d]] > u0:
        n = n + 1
    else:
        if n != 0:
            events_durations.append(n)
            events_id.append(W_max_daily.index[(d-1)-(n-1)])
            events_fd.append(W_max_daily.index[d-1])
            n = 0
durations = pd.Series(events_durations, index = events_fd, dtype = float, name = "durations of extreme maximum events")

# counting by month
counte_months = np.zeros(12, dtype = int)
for m in months:
    counte_months[m-1] = durations[durations.index.month == m].count()

# creating barplot
plt.figure(figsize = (16, 8))
barWidth = 0.45
plt.bar(months, counte_months, width = barWidth)
plt.xticks(months, months_names)
for m in months:
    plt.text(x = months[m-1]-0.08, y = counte_months[m-1]+0.5, s = counte_months[m-1], size = 12)
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Number of Extreme Maximum Events by Month from 1980-12-31 to 2019-12-31")
plt.show()

# counting by season
summere = ((durations.index.month == 12) & (durations.index.day >= 21)) + (durations.index.month == 1) + (durations.index.month == 2) + ((durations.index.month == 3) & (durations.index.day <= 20))
autumne = ((durations.index.month == 3) & (durations.index.day >= 21)) + (durations.index.month == 4) + (durations.index.month == 5) + ((durations.index.month == 6) & (durations.index.day <= 20))
wintere = ((durations.index.month == 6) & (durations.index.day >= 21)) + (durations.index.month == 7) + (durations.index.month == 8) + ((durations.index.month == 9) & (durations.index.day <= 20))
springe = ((durations.index.month == 9) & (durations.index.day >= 21)) + (durations.index.month == 10) + (durations.index.month == 11) + ((durations.index.month == 12) & (durations.index.day <= 20))
counte_seasons = np.zeros(4, dtype = int)
counte_seasons[0] = durations[summere].count()
counte_seasons[1] = durations[autumne].count()
counte_seasons[2] = durations[wintere].count()
counte_seasons[3] = durations[springe].count()

# creating barplot
plt.figure(figsize = (16, 8))
barWidth = 0.45
plt.bar(seasons, counte_seasons, width = barWidth)
plt.xticks(seasons, seasons_names)
for s in seasons:
    plt.text(x = seasons[s]-0.025, y = counte_seasons[s]+0.5, s = counte_seasons[s], size = 12)
plt.xlabel("Season")
plt.ylabel("Count")
plt.title("Number of Extreme Maximum Events by Season from 1980-12-31 to 2019-12-31")
plt.show()

# counting by year
counte_years = np.zeros(40, dtype = int)
for y in years:
    counte_years[y-1980] = durations[durations.index.year == y].count()
total_number_extreme_events = sum(counte_years)
print("From 1980-12-31 to 2019-12-31 there are {} extreme maximum events.".format(total_number_extreme_events))

print("\n")

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(counte_years, lags = 3, boxpierce = False, return_df = True)
print(lb_test)

print("\n")

# performing Mann-Kendall Trend (MK) Test
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(counte_years, alpha = 0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index = ["trend:", "p-value:"])
print(mk_output)

print("\n")

# creating barplot
plt.figure(figsize = (16, 8))
barWidth = 0.45
plt.bar(years, counte_years, width = barWidth)
plt.xticks(years, years, rotation = 90)
for y in years:
    plt.text(x = years[y-1980]-0.3, y = counte_years[y-1980]+0.5, s = counte_years[y-1980], size = 12)
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Number of Extreme Maximum Events from 1980-12-31 to 2019-12-31 and the Linear Trend Function")
x = (years - years[0]).reshape(-1, 1)
y = counte_years
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(years, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.show()

# accumulated number of extreme maximum events
accumulated_counte_numbers = np.zeros(40, dtype = int)
for j in range(0, 40):
    accumulated_counte_numbers[j] = sum(counte_years[k] for k in range(0, j+1))

# plotting the accumulated number of extreme maximum events
plt.figure(figsize = (16, 8))
plt.plot(years, accumulated_counte_numbers, linestyle = ":", marker = "o")
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Accumulated Number of Extreme Maximum Events by Year from 1980-12-31 to 2019-12-31")
plt.show()

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(durations, lags = 3, boxpierce = False, return_df = True)
print(lb_test)

print("\n")

# performing Mann-Kendall Trend (MK) Test
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(durations, alpha = 0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index = ["trend:", "p-value:"])
print(mk_output)

print("\n")

# plotting the durations of extreme maximum events
plt.figure(figsize = (16, 8))
st3 = plt.stem(durations.index, durations.values, use_line_collection = True)
plt.setp(st3[0], color = "orange", marker = "o", markersize = 2)
plt.setp(st3[1], color = "black", linestyle = "-")
plt.setp(st3[2], visible = False)
plt.xlabel("Date")
plt.ylabel("Duration (day)")
plt.title("Durations of Extreme Maximum Events from 1980-12-31 to 2019-12-31 and the Linear Trend Function")
x = ((durations.index - durations.index[0]).astype("timedelta64[h]")/24).values.reshape(-1, 1)
y = durations.values
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(durations.index, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.show()

# calculating the time elapsed between consecutive extreme maximum events
differences_events = []
for k in range(0, total_number_extreme_events-1):
    differences_events.append(events_id[k+1] - events_fd[k])
time_elapsed_events = pd.Series(differences_events)
time_elapsed_events = time_elapsed_events.rename("time elapsed between consecutive extreme maximum events")

# performing Ljung-Box Test
print("Results of the Ljung-Box Test\n")
lb_test = acorr_ljungbox(time_elapsed_events, lags = 3, boxpierce = False, return_df = True)
print(lb_test)

print("\n")

# performing Mann-Kendall Trend (MK) Test
print("Results of Mann-Kendall Trend Test\n")
mk_test = mk.original_test(time_elapsed_events.astype("timedelta64[h]")/24, alpha = 0.05)
mk_output = pd.Series([mk_test[0], mk_test[2]], index = ["trend:", "p-value:"])
print(mk_output)

#print("\n")

# plotting the time elapsed between consecutive extreme maximum events
plt.figure(figsize = (16, 8))
st4 = plt.stem(time_elapsed_events.index, time_elapsed_events.astype("timedelta64[h]").values/24, use_line_collection = True)
plt.setp(st4[0], color = "orange", marker = "o", markersize = 2)
plt.setp(st4[1], color = "black", linestyle = "-")
plt.setp(st4[2], visible = False)
plt.xlabel("Gap Number")
plt.ylabel("Time Elapsed (day)")
plt.title("Time Elapsed Between Consecutive Extreme Maximum Events and the Linear Trend Function")
x = (time_elapsed_events.index - time_elapsed_events.index[0]).values.reshape(-1, 1)
y = (time_elapsed_events.astype("timedelta64[h]")/24).values
model = linear_model.LinearRegression().fit(x, y)
y_fit = model.predict(x)
print("Results of the Linear Regression\n")
lm_output = pd.Series([model.coef_, model.intercept_, mean_squared_error(y, y_fit), r2_score(y, y_fit)], index = ["slope:", "intercept:", "mean squared error:", "r2 score:"])
print(lm_output)
plt.plot(time_elapsed_events.index, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
plt.legend(loc = "best")
plt.show()

