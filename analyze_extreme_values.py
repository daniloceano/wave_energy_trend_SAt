import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import typing
import matplotlib
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from glob import glob
from pyextremes.tuning.threshold_selection import get_default_thresholds
from pyextremes.tuning.threshold_selection import _calculate_modified_parameters
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from thresholdmodeling import thresh_modeling

# Constants
FIGS_DIR = "./figures/"
PROCESSED_DATA_DIR = "./processed_data/"
VARIABLE = "swh"
GRAY = '#343a40'
RED = '#bf0603'
BLUE = '#0077b6'
ALPHA = 0.05

# Function Definitions
def MRL(sample, alpha):
    """
    Calculate Mean Residual Life (MRL) and Confidence Intervals.

    Parameters
    ----------
    sample : array_like
        The sample data.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object of the plot.
    """
    step = np.quantile(sample, .995) / 60
    threshold = np.arange(0, max(sample), step=step)
    z_inverse = norm.ppf(1 - (alpha / 2))

    mrl_array = []
    CImrl = []

    for u in threshold:
        excess = [data - u for data in sample if data > u]
        mrl_array.append(np.mean(excess))
        std_loop = np.std(excess)
        CImrl.append(z_inverse * std_loop / (len(excess)**0.5))

    CI_Low = [mrl - ci for mrl, ci in zip(mrl_array, CImrl)]
    CI_High = [mrl + ci for mrl, ci in zip(mrl_array, CImrl)]

    fig, ax = plt.subplots()
    ax.plot(threshold, mrl_array)
    ax.fill_between(threshold, CI_Low, CI_High, alpha=0.4)
    ax.set_xlabel('u')
    ax.set_ylabel('Mean Excesses')
    ax.set_title('Mean Residual Life Plot')
    return fig, ax

def get_parameter_stability(ts, thresholds=None, r="24H", extremes_type="high", alpha=None, n_samples=100, progress=False):
    """
    Generate parameter stability results for given threshold values.

    Parameters
    ----------
    ts : pd.Series
        The time series data.
    thresholds : array_like, optional
        Array of threshold values.
    r : str or pd.Timedelta, default "24H"
        Time resolution.
    extremes_type : str, default "high"
        Type of extremes to analyze ("high" or "low").
    alpha : float, optional
        Alpha value for confidence interval.
    n_samples : int, default 100
        Number of samples for stability analysis.
    progress : bool, default False
        Show progress bar if True.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with shape and scale parameter values and confidence intervals.
    """
    try:
        import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        if progress:
            raise ImportError(
                "'tqdm' package is required to display a progress bar"
            ) from error

    # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

        # Get default thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(
            ts=ts,
            extremes_type=extremes_type,
            num=100,
        )

    # List of unique seeds - ensures same seed is not reused across sub-processes
    seeds: typing.List[int] = []

    def _input_generator() -> (
        typing.Generator[
            typing.Tuple[
                pd.Series,  # ts (time series)
                str,  # extremes_type
                float,  # threshold
                typing.Union[str, pd.Timedelta],  # r
                typing.Optional[float],  # alpha
                int,  # n_samples
                int,  # seed
            ],
            None,
            None,
        ]
    ):
        for threshold in thresholds:
            seed = np.random.randint(low=0, high=1e6, size=None)
            while seed in seeds:
                seed = np.random.randint(low=0, high=1e6, size=None)
            seeds.append(seed)
            yield (ts, extremes_type, threshold, r, alpha, n_samples, seed)

    iterable = (
        tqdm.tqdm(
            _input_generator(),
            desc="calculating stability parameters",
            total=len(thresholds),
            smoothing=0,
        )
        if progress
        else _input_generator()
    )

    cpu_count = os.cpu_count() or 1
    # Compute results
    if cpu_count > 1:
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            _results = list(pool.imap(_calculate_modified_parameters, iterable))
    else:
        _results = []
        for args in iterable:
            _results.append(_calculate_modified_parameters(args))
    results = (
        pd.DataFrame(data=_results).set_index("threshold").sort_index(ascending=True)
    )

    # Return the results dataframe
    return results

def plot_stability_results(results, axes=None, figsize=(8, 5), alpha=None):
    """
    Plot shape and scale parameter stability using results from parameter stability analysis.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame with shape and scale parameter values and confidence intervals.
    axes : tuple, optional
        Tuple with matplotlib Axes for shape and scale values.
    figsize : tuple, optional
        Figure size.
    alpha : float, optional
        Confidence interval width.

    Returns
    -------
    ax_shape : matplotlib.axes.Axes
        Axes with shape parameter values.
    ax_scale : matplotlib.axes.Axes
        Axes with scale parameter values.
    """
    if axes is None:
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=96)

        # Create gridspec
        gs = matplotlib.gridspec.GridSpec(
            nrows=2,
            ncols=1,
            wspace=0.1,
            hspace=0.1,
            width_ratios=[1],
            height_ratios=[1, 1],
        )

        # Create and configure axes
        ax_shape = fig.add_subplot(gs[0, 0])
        ax_scale = fig.add_subplot(gs[1, 0])
    else:
        fig = None
        ax_shape, ax_scale = axes

    # Plot central estimates of shape and modified scale parameters
    ax_shape.plot(results.loc[:, "shape"], ls="-", color=GRAY, lw=2, zorder=15)
    ax_scale.plot(results.index, results.loc[:, "scale"], ls="-", color=GRAY, lw=2, zorder=15)

    # Plot confidence bounds
    if alpha is not None:
        for ax, parameter in [(ax_shape, "shape"), (ax_scale, "scale")]:
            for ci in ["lower", "upper"]:
                ax.plot(results.index, results.loc[:, f"{parameter}_ci_{ci}"], color=BLUE,
                        lw=1, ls="--", zorder=10)
            ax.fill_between(results.index, results.loc[:, f"{parameter}_ci_lower"], results.loc[:, f"{parameter}_ci_upper"],
                facecolor=BLUE, edgecolor="None", alpha=0.5, zorder=5)

    if fig is not None:
        # Configure axes
        ax_shape.tick_params(axis="x", which="both", labelbottom=False, length=0)
        ax_scale.set_xlim(ax_shape.get_xlim())

    # Label axes
    ax_shape.set_ylabel(r"Shape, $\xi$")
    ax_scale.set_ylabel(r"Modified scale, $\sigma^*$")
    if fig is not None:
        ax_scale.set_xlabel("Threshold")

    return ax_shape, ax_scale

def load_time_series(variable):
    """
    Load time series data for the given variable.

    Parameters
    ----------
    variable : str
        Variable name to load time series for.

    Returns
    -------
    data : pd.DataFrame
        Loaded time series data.
    """
    data_file = glob(os.path.join(PROCESSED_DATA_DIR, f"W_{variable}_24.12S_45.71W_1980-2022.csv"))[0]
    data = pd.read_csv(data_file, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

def perform_analysis():
    """
    Main analysis function to perform the time series analysis and plotting.
    """
    importr("base")
    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)

    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["black"])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    importr("base")
    utils = importr("utils")
    utils.chooseCRANmirror(ind = 1)

    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["black"])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False

    # Import W
    W = load_time_series(VARIABLE)

    # constructing the maximum daily normalized deviation time series from the time series W
    W_max_daily = W.resample('D').max()
    W_max_daily = W_max_daily.rename(columns={'normalized deviation': 'maximum daily normalized deviation'})

    # Observação: Nós poderíamos ter construído a série do desvio máximo diário com o método .resample:
    # W_max_daily = W.resample("24H").max()
    # Entretanto, o índice seria modificado: disposto dia-a-dia, sem o registro da hora. Isso não é bom para nosso problema, pois
    # após toda a análise a ser feita, queremos ser capazes de voltar à série temporal original X e ver quais são os valores de
    # temperatura dos extremos.

    lag_acf = acf(W_max_daily, nlags = 192, fft = False)
    plt.close("all")
    plt.figure(figsize = (16, 8))
    plt.plot(lag_acf)
    plt.xlabel("Lag (day)")
    plt.ylabel("Autocorrelation")
    plt.axhline(y = 0, linestyle="--", color=GRAY)
    plt.axhline(y = -1.96/np.sqrt(len(W_max_daily)), linestyle="--", color=GRAY)
    plt.axhline(y = 1.96/np.sqrt(len(W_max_daily)), linestyle="--", color=GRAY)
    plt.title("Autocorrelation Function for the Time Series $W_{\mathrm{max\_daily}}$")
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_function_W_max_daily_{VARIABLE}.png"))

    # performing Ljung-Box Test
    print("Results of the Ljung-Box Test\n")
    lb_test = acorr_ljungbox(W_max_daily, lags=192, boxpierce=False, return_df=True)
    print(lb_test)
    # Interpretation of Ljung-Box test result
    p_value = lb_test['lb_pvalue']
    plt.close("all")
    fig = plt.figure(figsize=(8,8))
    plt.plot(p_value, color=GRAY)
    plt.axhline(y = 0.05, linestyle="--", color=RED)
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.title("Ljung-Box Test for $W_{\mathrm{max\_daily}}$")
    plt.savefig(os.path.join(FIGS_DIR, f"lb_test_W_max_daily_{VARIABLE}.png"))

    # chosen threshold
    u0 = W_max_daily.quantile(0.97)['maximum daily normalized deviation']

    # plotting the mean excess (mean residual life) function with a confidence level of 5%
    plt.close("all")
    fig, ax = MRL(W_max_daily['maximum daily normalized deviation'], 0.05)
    plt.axvline(W_max_daily.quantile(0.97)['maximum daily normalized deviation'], color=RED, linestyle="--",
                label=f"Threshold Line $u_{0}$ = q0.97 ({round(u0,2)})")
    plt.legend(loc = "best")
    plt.savefig(os.path.join(FIGS_DIR, f"mean_excess_function_W_max_daily_{VARIABLE}.png"))

    # plotting the shape and modified scale parameters stability with a confidence level of 5%
    plt.close("all")
    results = get_parameter_stability(W_max_daily['maximum daily normalized deviation'], alpha=0.05)
    ax_shape, ax_scale = plot_stability_results(results, alpha=0.05)
    plt.savefig(os.path.join(FIGS_DIR, f"parameter_stability_plot_W_max_daily_{VARIABLE}.png"))

    # plotting the time series W_max_daily and the threshold line
    plt.close("all")
    plt.figure(figsize=(12, 8))
    plt.plot(W_max_daily, label="Time Series $W_{\mathrm{max\_daily}}$", color=GRAY)
    plt.axhline(float(u0), color=RED, linestyle="--", label=f"Threshold Line $u_{0}$ = q0.97 ({round(u0,2)})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Deviations")
    plt.legend(loc = "best")
    plt.title("Time Series $W_{\mathrm{max\_daily}}$ and the Threshold Line")
    plt.savefig(os.path.join(FIGS_DIR, f"time_series_W_max_daily_{VARIABLE}.png"))

    # exceedances over the threshold series
    E = W_max_daily[W_max_daily['maximum daily normalized deviation'] > u0]['maximum daily normalized deviation'] - u0
    lag_acf = acf(E, nlags = 192, fft = False)
    plt.figure(figsize = (16, 8))
    plt.plot(lag_acf)
    plt.xlabel("Lag (day)")
    plt.ylabel("Autocorrelation")
    plt.axhline(y = 0, linestyle = "--", color = "gray")
    plt.axhline(y = -1.96/np.sqrt(len(E)), linestyle = "--", color = "gray")
    plt.axhline(y = 1.96/np.sqrt(len(E)), linestyle = "--", color = "gray")
    plt.title("Autocorrelation Function for Exceedances Over the Threshold Series $E$")
    plt.savefig(os.path.join(FIGS_DIR, f"autocorrelation_function_E_{VARIABLE}.png"))

    # performing Ljung-Box Test
    print("Results of the Ljung-Box Test\n")
    lb_test = acorr_ljungbox(E, lags = 192, boxpierce = False, return_df = True)
    print(lb_test)
    p_value = lb_test['lb_pvalue']
    plt.close("all")
    plt.figure(figsize = (8, 8))
    plt.plot(p_value, color=GRAY)
    plt.axhline(y=0.05, linestyle="--", color=RED)
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.title("Ljung-Box Test for $E$")
    plt.savefig(os.path.join(FIGS_DIR, f"lb_test_E_{VARIABLE}.png"))

    # fitting the time series W_max_daily data to a GPD model with the chosen threshold u0
    u0_r = FloatVector([u0])
    thresh_modeling.gpdfit(W_max_daily, u0_r[0], "mle")

    # model checking (empirical versus model): plotting quantile-quantile, probability-probability and cumulative-cumulative 
    # functions with the chosen threshold u0 and with a confidence level of 5%
    plt.close("all")
    fig_qq = thresh_modeling.qqplot(np.ravel(W_max_daily["maximum daily normalized deviation"]), u0_r[0], "mle", 0.05)
    fig_qq.savefig(os.path.join(FIGS_DIR, f"qqplot_W_max_daily_{VARIABLE}.png"))

    plt.close("all")
    fig_pp = thresh_modeling.ppplot(np.ravel(W_max_daily["maximum daily normalized deviation"]), u0_r[0], "mle", 0.05)
    fig_pp.savefig(os.path.join(FIGS_DIR, f"ppplot_W_max_daily_{VARIABLE}.png"))

    plt.close("all")
    fig_cdf = thresh_modeling.gpdcdf(W_max_daily["maximum daily normalized deviation"], u0_r[0], "mle", 0.05)
    fig_cdf.savefig(os.path.join(FIGS_DIR, f"gpdcdf_W_max_daily_{VARIABLE}.png"))

if __name__ == '__main__':
    perform_analysis()