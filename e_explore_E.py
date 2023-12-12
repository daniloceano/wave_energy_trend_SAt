# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    e_explore_E.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/11 14:32:51 by daniloceano       #+#    #+#              #
#    Updated: 2023/12/12 08:16:20 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
The 'e_explore_E.py' module is part of the South Atlantic waves trend analysis project. 
It focuses on examining time series data for extreme events using Extreme Value Theory (EVT). 
The module analyzes how often these events happen, when they occur, and their patterns over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from a_get_unseazon_series import load_data
from d_peak_over_threshold import load_time_series, read_u0_from_file
from c_get_exceedance_E_series import load_time_series as load_W

from statsmodels.stats.diagnostic import acorr_ljungbox
import pymannkendall as mk
import os

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

def perform_mann_kendall_test(data, filename):
    """
    Perform Mann-Kendall test and save the results.

    Args:
        data (pandas.DataFrame): Data to test.
        filename (str): Path to save the test results.
    """
    mk_test = mk.original_test(data, alpha=0.05)
    mk_output = pd.Series([mk_test[0], mk_test[2]], index=["trend:", "p-value:"])
    with open(filename, 'w') as file:
        file.write(mk_output.to_string())

# Function to determine the season for a given date
def get_season(date):
    """
    Determine the season for a given date.
    Args:
        date (datetime): Date for which to determine the season.
    Returns:
        str: Season name.
    """
    month, day = date.month, date.day
    if (month == 12 and day >= 21) or (1 <= month <= 2) or (month == 3 and day <= 20):
        return 'Summer'
    elif (month == 3 and day >= 21) or (4 <= month <= 5) or (month == 6 and day <= 20):
        return 'Autumn'
    elif (month == 6 and day >= 21) or (7 <= month <= 8) or (month == 9 and day <= 20):
        return 'Winter'
    else:
        return 'Spring'

def main():
    # Define file paths
    lat, lon = np.round(float(POINT_LAT), 2), np.round(float(POINT_LON), 2)
    lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
    lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
    filename = f"E_{VARIABLE}_{lat_str}_{lon_str}"

    # Load data
    data_file = os.path.join(PROCESSED_DATA_DIR, f"E_{VARIABLE}_{lat_str}_{lon_str}_{START_YEAR}-{END_YEAR}.csv")
    E = load_time_series(data_file)

    # Import original series
    # Load data
    ds = load_data("Hs*nc")
    lat, lon = np.round(float(ds.latitude), 2), np.round(float(ds.longitude), 2)
    lat_str = f"{np.abs(lat)}S" if lat < 0 else f"{lat}N"
    lon_str = f"{np.abs(lon)}W" if lon < 0 else f"{lon}E"
    X = ds.squeeze(['longitude', 'latitude']).to_dataframe().drop(columns=["latitude", "longitude"])

    # Create a new Series by selecting values from 'X' at the indices where 'E' has data
    X_e = X.loc[E.index]['swh']
    # Rename the series for clarity
    X_e.rename(f"extreme maximum {VARIABLE}", inplace=True)
    print(f"From {X.index[0].date()} to {X.index[-1].date()} at {POINT_LAT}, {POINT_LON}, there are {len(X_e)} extreme maximum values of {VARIABLE}")
    csv_filename = os.path.join(PROCESSED_DATA_DIR, f"XE_{VARIABLE}_{lat_str}_{lon_str}_{START_YEAR}-{END_YEAR}.csv")
    X_e.to_csv(csv_filename)
    print(f"Extreme maximum values for {VARIABLE} at {lat}, {lon} saved to {csv_filename}")

    # Month-wise boxplots for the extreme maximum values
    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.close("all")
    plt.figure(figsize = (16, 8))
    bp = plt.boxplot([X_e[X_e.index.month == m] for m in range(1, 13)], labels = months_names, vert = True)
    plt.setp(bp["medians"], color = RED, linewidth=3)
    plt.setp(bp["boxes"], color = GRAY)
    plt.setp(bp["whiskers"], color = GRAY, linestyle = "--")
    plt.setp(bp["fliers"], color = GRAY, marker = "+")
    plt.xlabel("Month")
    plt.ylabel(VARIABLE.upper())
    plt.title(f"Month-wise Boxplots for the Extreme Maximum {VARIABLE.upper()} Series from {X.index[0].date()} to {X.index[-1].date()}")
    boxplot_filename = os.path.join(FIGS_DIR, f"boxplots_monthly_XE_{filename}.png")
    plt.savefig(boxplot_filename)
    print(f"Boxplots for extreme maximum values (XE) for {VARIABLE} at {lat}, {lon} saved to {boxplot_filename}")

    # Month-wise boxplots for the exceedances over the threshold series
    plt.close("all")
    plt.figure(figsize = (16, 8))
    bp = plt.boxplot([E[E.index.month == m]['maximum daily normalized deviation'] for m in range(1, 13)], labels = months_names, vert = True)
    plt.setp(bp["medians"], color = RED, linewidth=3)
    plt.setp(bp["boxes"], color = GRAY)
    plt.setp(bp["whiskers"], color = GRAY, linestyle = "--")
    plt.setp(bp["fliers"], color = GRAY, marker = "+")
    plt.xlabel("Month")
    plt.ylabel("Excesses")
    plt.title("Month-wise Boxplots for the Exceedances Over the Threshold Series")
    boxplot_filename = os.path.join(FIGS_DIR, f"boxplots_montlhy_{filename}.png")
    plt.savefig(boxplot_filename)
    print(f"Boxplots for exceedances (E) for {VARIABLE} at {lat}, {lon} saved to {boxplot_filename}")

    # Year-wise boxplots for the exceedances over the threshold series
    years = np.arange(1980, 2020, dtype = int)
    plt.close("all")
    plt.figure(figsize = (16, 8))
    bp = plt.boxplot([E[E.index.year == y]['maximum daily normalized deviation'] for y in years], labels = years, vert = True)
    plt.xticks(rotation = 90)
    plt.setp(bp["medians"], color = RED, linewidth=3)
    plt.setp(bp["boxes"], color = GRAY)
    plt.setp(bp["whiskers"], color = GRAY, linestyle = "--")
    plt.setp(bp["fliers"], color = GRAY, marker = "+")
    plt.xlabel("Year")
    plt.ylabel("Excesses")
    plt.title("Year-wise Boxplots for the Exceedances Over the Threshold Series")
    boxplot_filename = os.path.join(FIGS_DIR, f"boxplots_yearly_{filename}.png")
    plt.savefig(boxplot_filename)
    print(f"Boxplots for exceedances (E) for {VARIABLE} at {lat}, {lon} saved to {boxplot_filename}")

    # Counting by month
    months = np.arange(1, 13, dtype = int)
    countd_months = np.zeros(12, dtype = int)
    for m in months:
        countd_months[m-1] = E[E.index.month == m].count()
    plt.close("all")
    plt.figure(figsize = (16, 8))
    barWidth = 0.45
    plt.bar(months, countd_months, width=barWidth, color=GRAY)
    plt.xticks(months, months_names)
    for m in months:
        plt.text(x=months[m-1]-0.08, y=countd_months[m-1]+0.5, s = countd_months[m-1], size = 12)
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.title(f"Number of Extreme Maximum Daily {VARIABLE} by Month from {X.index[0].date()} to {X.index[-1].date()}")
    barplot_filename = os.path.join(FIGS_DIR, f"barplot_countd_months_{filename}.png")
    plt.savefig(barplot_filename)
    print(f"Barplot for count of exceedances (E) for {VARIABLE} at {lat}, {lon} saved to {barplot_filename}")

    # counting by season
    E.index = pd.to_datetime(E.index)
    E['Season'] = E.index.map(get_season)
    # Count the number of occurrences for each season
    countd_seasons = E['Season'].value_counts()
    # Sort the counts by season order
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    countd_seasons = countd_seasons.reindex(season_order)
    # Define colors for each season
    season_colors = {'Summer': RED, 'Autumn': ORANGE, 'Winter': BLUE, 'Spring': GREEN}
    # Plotting
    plt.close("all")
    plt.figure(figsize=(16, 8))
    barWidth = 0.45
    for i, season in enumerate(season_order):
        plt.bar(i, countd_seasons[season], width=barWidth, color=season_colors[season], alpha=0.8)
        plt.text(x=i - 0.025, y=countd_seasons[season] + 0.5, s=countd_seasons[season], size=12)
    plt.xticks(range(len(season_order)), season_order)
    plt.xlabel("Season")
    plt.ylabel("Count")
    plt.title(f"Number of Extreme Maximum Daily {VARIABLE} by Season from {X.index[0].date()} to {X.index[-1].date()}")
    barplot_filename = os.path.join(FIGS_DIR, f"barplot_countd_seasons_{filename}.png")
    plt.savefig(barplot_filename)
    print(f"Barplot for count of exceedances (E) for {VARIABLE} at {lat}, {lon} saved to {barplot_filename}")

    # counting by year
    countd_years = np.zeros(40, dtype = int)
    for y in years:
        countd_years[y-1980] = E[E.index.year == y]['maximum daily normalized deviation'].count()
    total_number_extreme_days = sum(countd_years) # = len(E)
    print(f"From {X.index[0].date()} to {X.index[-1].date()} at {lat}, {lon}, there are {total_number_extreme_days} daily extreme values of {VARIABLE}.")

    # performing Ljung-Box Test
    print("Results of the Ljung-Box Test\n")
    lb_test = acorr_ljungbox(countd_years, lags=3, boxpierce=False, return_df=True)
    p_value = lb_test['lb_pvalue']
    plt.close("all")
    plt.figure(figsize=(8,8))
    plt.plot(p_value, color=GRAY)
    plt.axhline(y=0.05, linestyle="--", color=RED)
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.title("Ljung-Box Test for $W$")
    lb_test_filename = os.path.join(FIGS_DIR, f"lb_test_{filename}.png")
    plt.savefig(lb_test_filename)
    print(f"Ljung-Box Test for {VARIABLE} at {lat}, {lon} saved to {lb_test_filename}")

    # performing Mann-Kendall Trend (MK) Test
    mk_test_file = os.path.join(PROCESSED_DATA_DIR, f"kendall-test_countd_years_{filename}.txt")
    perform_mann_kendall_test(countd_years, mk_test_file)
    print(f"Mann-Kendall Test for yearly count of exceedances (E) of {VARIABLE} at {lat}, {lon} saved to {mk_test_file}")

    # creating barplot
    plt.close("all")
    plt.figure(figsize = (16, 8))
    barWidth = 0.45
    plt.bar(years, countd_years, width=barWidth, color=GRAY, alpha=0.8)
    plt.xticks(years, years, rotation = 90)
    for y in years:
        plt.text(x = years[y-1980]-0.3, y = countd_years[y-1980]+0.5, s = countd_years[y-1980], size = 12)
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title(f"Number of extreme maximum daily {VARIABLE} by year from {X.index[0].date()} to {X.index[-1].date()} and the Linear Trend Function")
    x = (years - years[0]).reshape(-1, 1)
    y = countd_years
    model = linear_model.LinearRegression().fit(x, y)
    y_fit = model.predict(x)
    plt.plot(years, y_fit, color=RED, label="Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
    plt.legend(loc = "best")
    barplot_filename = os.path.join(FIGS_DIR, f"barplot_countd_years_trand_{filename}.png")
    plt.savefig(barplot_filename)
    print(f"Barplot for count of yearly exceedances (E) and trend for {VARIABLE} at {lat}, {lon} saved to {barplot_filename}")

    # accumulated number of extreme maximum events
    accumulated_counte_numbers = np.zeros(40, dtype = int)
    for j in range(0, 40):
        accumulated_counte_numbers[j] = sum(countd_years[k] for k in range(0, j+1))
    plt.close("all")
    plt.figure(figsize = (16, 8))
    plt.plot(years, accumulated_counte_numbers, linestyle=":", marker="o", color=GRAY)
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title(f"Accumulated Number of Extreme Maximum Events by Year from {X.index[0].date()} to {X.index[-1].date()}")
    accumulated_counte_numbers_filename = os.path.join(FIGS_DIR, f"accumulated_countd_numbers_{filename}.png")
    plt.savefig(accumulated_counte_numbers_filename)
    print(f"Accumulated number of yearly exceedances (E) for {VARIABLE} at {lat}, {lon} saved to {accumulated_counte_numbers_filename}")

    # Load W and u0
    W = load_W(VARIABLE, lat_str, lon_str, START_YEAR, END_YEAR)
    W_max_daily = W.resample("D").max()
    u0_file = os.path.join(PROCESSED_DATA_DIR, f"u0_{VARIABLE}_{lat_str}_{lon_str}_{START_YEAR}-{END_YEAR}.txt")
    u0 = read_u0_from_file(u0_file)

    # Improved event identification and duration calculation
    events = []
    current_event_start = None

    for day, row in W_max_daily.iterrows():
        if row['normalized deviation'] > u0:
            if current_event_start is None:
                current_event_start = day
        else:
            if current_event_start is not None:
                event_duration = (day - current_event_start).days
                events.append({'start_date': current_event_start, 'end_date': day - pd.Timedelta(days=1), 'duration': event_duration})
                current_event_start = None

    # Create a Series for durations
    durations = pd.Series([event['duration'] for event in events], index=[event['end_date'] for event in events], dtype=float, name="durations of extreme maximum events")

    # performing Ljung-Box Test
    print("Results of the Ljung-Box Test\n")
    lb_test = acorr_ljungbox(durations, lags = 3, boxpierce = False, return_df = True)
    p_value = lb_test['lb_pvalue']
    plt.close("all")
    plt.figure(figsize = (8, 8))
    plt.plot(p_value, color=GRAY)
    plt.axhline(y=0.05, linestyle="--", color=RED)
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.title("Ljung-Box Test for exceedances (E) durations")
    lb_test_filename = os.path.join(FIGS_DIR, f"lb_test_durations_{filename}.png")
    plt.savefig(lb_test_filename)
    print(f"Ljung-Box Test for exceedances (E) durations for {VARIABLE} at {lat}, {lon} saved to {lb_test_filename}")

    # performing Mann-Kendall Trend (MK) Test
    mk_test_file = os.path.join(PROCESSED_DATA_DIR, f"kendall-test_durations_{filename}.txt")
    perform_mann_kendall_test(countd_years, mk_test_file)
    print(f"Mann-Kendall Test for counts of yearly exceedances (E) of {VARIABLE} at {lat}, {lon} saved to {mk_test_file}")

    # plotting the durations of extreme maximum events
    plt.close("all")
    plt.figure(figsize = (16, 8))
    st3 = plt.stem(durations.index, durations.values)
    plt.setp(st3[0], color=ORANGE, marker="o", markersize=2)
    plt.setp(st3[1], color=GRAY, linestyle="-")
    plt.setp(st3[2], visible = False)
    plt.xlabel("Date")
    plt.ylabel("Duration (day)")
    plt.title("Durations of Extreme Maximum Events from 1980-12-31 to 2019-12-31 and the Linear Trend Function")
    x = (durations.index - durations.index[0]).days.values.reshape(-1, 1)
    y = durations.values
    model = linear_model.LinearRegression().fit(x, y)
    y_fit = model.predict(x)
    plt.plot(durations.index, y_fit, color=RED, label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
    plt.legend(loc = "best")
    durations_filename = os.path.join(FIGS_DIR, f"trend_duration_{filename}.png")
    plt.savefig(durations_filename)
    print(f"Durations of extreme maximum events for {VARIABLE} at {lat}, {lon} saved to {durations_filename}")

    # calculating the time elapsed between consecutive extreme maximum events
    total_number_extreme_events = len(countd_years)
    differences_events = []
    for k in range(0, total_number_extreme_events-1):
        differences_events.append(events[k+1]['duration'] - events[k]['duration'])
    time_elapsed_events = pd.Series(differences_events)
    time_elapsed_events = time_elapsed_events.rename("time elapsed between consecutive extreme maximum events")

    # performing Ljung-Box Test
    print("Results of the Ljung-Box Test\n")
    lb_test = acorr_ljungbox(time_elapsed_events, lags = 3, boxpierce = False, return_df = True)
    p_value = lb_test['lb_pvalue']
    plt.close("all")
    plt.figure(figsize = (8, 8))
    plt.plot(p_value, color=GRAY)
    plt.axhline(y=0.05, linestyle="--", color=RED)
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.title("Ljung-Box Test for time elapsed between consecutive extreme maximum events")
    lb_test_filename = os.path.join(FIGS_DIR, f"lb_test_time_elapsed_{filename}.png")
    plt.savefig(lb_test_filename)
    print(f"Ljung-Box Test for time elapsed between consecutive extreme maximum events for {VARIABLE} at {lat}, {lon} saved to {lb_test_filename}")

    # performing Mann-Kendall Trend (MK) Test
    mk_test_file = os.path.join(PROCESSED_DATA_DIR, f"kendall-test_time_elapsed_{filename}.txt")
    perform_mann_kendall_test(time_elapsed_events, mk_test_file)
    print(f"Mann-Kendall Test for time elapsed between consecutive extreme maximum events exceedances (E) of {VARIABLE} at {lat}, {lon} saved to {mk_test_file}")

    # plotting the time elapsed between consecutive extreme maximum events
    plt.close("all")
    plt.figure(figsize = (16, 8))
    st4 = plt.stem(time_elapsed_events.index, time_elapsed_events.astype("timedelta64[h]").values/24)
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
    plt.plot(time_elapsed_events.index, y_fit, color = "red", label = "Line $y = $" + str(model.coef_) + " $x +$" + str(model.intercept_))
    plt.legend(loc = "best")
    time_elapsed_filename = os.path.join(FIGS_DIR, f"trend_time_elapsed_{filename}.png")
    plt.savefig(time_elapsed_filename)
    print(f"Time elapsed between consecutive extreme maximum events for {VARIABLE} at {lat}, {lon} saved to {time_elapsed_filename}")

if __name__ == "__main__":
    main()
