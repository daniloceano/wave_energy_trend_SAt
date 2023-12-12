# South Atlantic waves trend analysis

## Overview
This project focuses on the analysis of time series data, with a particular emphasis on environmental and climate research. It comprises three distinct modules, each designed for specific aspects of time series analysis, from data preprocessing to advanced statistical evaluations.

## Modules

### Module 1: `a_get_unseazon_series.py`
- **Purpose**: Handles the deseasonalization and detrending of time series data, preparing it for further analysis.
- **Key Functionalities**:
  - Deseasonalizing and detrending time series data to remove seasonal patterns and long-term trends.
  - Calculating moving statistics for short-term fluctuation smoothing.
  - Generating plots to inspect the original, deseasonalized, and detrended time series.

### Module 2: `b_W_stationarity_analysis.py`
- **Purpose**: Focused on assessing the statistical properties of detrended data, specifically its stationarity and independence.
- **Key Functionalities**:
  - Conducting the Augmented Dickey-Fuller test to determine the stationarity of the time series.
  - Performing the Ljung–Box Test to evaluate the independence of the time series data.

### Module 3: `c_get_exceedance_series.py`
- **Purpose**: Concentrates on identifying an appropriate threshold for Peak Over Threshold (POT) analysis within extreme value theory.
- **Key Functionalities**:
  - Calculating Mean Residual Life (MRL) to aid in selecting a suitable threshold for extreme value analysis.
  - Assessing the stability of shape and scale parameters in the Generalized Pareto Distribution (GPD).
  - Determines the threshold that will be used in the POT analysis

### Module 4: `d_peak_over_threshold.py`
- **Purpose**: Conducts Peak Over Threshold (POT) analysis, a key aspect of extreme value theory.
- **Key Functionalities**:
  - Threshold Reading Retrieves a predefined threshold value (u0) from a text file.
  - Histogram with Fitted GPD Generates and saves histograms of exceedances, overlaying a fitted Generalized Pareto Distribution (GPD).
  - Mann-Kendall Trend Test Implements the Mann-Kendall test to detect trends in the exceedance data.
  - Linear Trend Analysis Visualizes and quantifies changes in extremes by plotting and saving the linear trend of the exceedance data.

### Module 5: `e_explore_E.py`
- **Purpose**: This module is designed for in-depth exploration and analysis of exceedances over the threshold in time series data. It focuses on understanding the distribution, frequency, and trends of extreme events.
- **Key Functionalities**:
  - Seasonal and Monthly Analysis: Determines the distribution of extreme events across different months and seasons, using boxplots and bar charts for visualization.
  - Trend Analysis: Employs the Mann-Kendall test to detect trends in the frequency of extreme events over time.
  - Event Duration Analysis: Calculates and visualizes the duration of extreme events, providing insights into the persistence of extreme conditions.
  - Time Elapsed Analysis: Analyzes the time elapsed between consecutive extreme events, helping to understand the frequency and clustering of such events.
  - Ljung-Box Test: Performs statistical tests to assess the autocorrelation in the duration and time elapsed between extreme events.

### Module 6: `f_return_period.py`
- **Purpose**: This module is designed to calculate and visualize the return levels of extreme events in time series data. It focuses on estimating the intensity of events that are likely to occur once every N years, a key concept in extreme value theory and risk assessment.
- **Key Functionalities**:
  - Threshold Application: Reads a predefined threshold value (u0) from a file, which is used to identify significant exceedances in the time series.
  - Return Level Calculation: Utilizes the thresh_modeling library to calculate the N-year return level for the time series.
  - Visualization: Generates a plot of the return level, offering a visual representation of the risk associated with extreme events over a specified period.