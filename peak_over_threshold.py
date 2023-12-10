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