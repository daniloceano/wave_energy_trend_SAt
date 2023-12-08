# South Atlantic waves trend analysis

## Overview
This project focuses on the analysis of time series data, with a particular emphasis on environmental and climate research. It comprises three distinct modules, each designed for specific aspects of time series analysis, from data preprocessing to advanced statistical evaluations.

## Modules

### Module 1: `get_unseazon_series.py`
- **Purpose**: Handles the deseasonalization and detrending of time series data, preparing it for further analysis.
- **Key Functionalities**:
  - Deseasonalizing and detrending time series data to remove seasonal patterns and long-term trends.
  - Calculating moving statistics for short-term fluctuation smoothing.
  - Generating plots to inspect the original, deseasonalized, and detrended time series.

### Module 2: `stationarity_analysis.py`
- **Purpose**: Focused on assessing the statistical properties of detrended data, specifically its stationarity and independence.
- **Key Functionalities**:
  - Conducting the Augmented Dickey-Fuller test to determine the stationarity of the time series.
  - Performing the Ljung–Box Test to evaluate the independence of the time series data.

### Module 3: `analyze_extreme_values.py`
- **Purpose**: Concentrates on identifying an appropriate threshold for Peak Over Threshold (POT) analysis within extreme value theory.
- **Key Functionalities**:
  - Calculating Mean Residual Life (MRL) to aid in selecting a suitable threshold for extreme value analysis.
  - Assessing the stability of shape and scale parameters in the Generalized Pareto Distribution (GPD).
