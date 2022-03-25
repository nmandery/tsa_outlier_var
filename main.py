# based on
# * https://www.analyticsvidhya.com/blog/2021/08/multivariate-time-series-anomaly-detection-using-var-model/

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR

df = pd.read_csv("data/walking-stairs.2.csv")
df.sort_values(by=["time_s"], inplace=True)
del df["lin_acc_z"]
del df["lin_acc_y"]
print(df)


def vector_autoregression_anomalies(df: pd.DataFrame, time_column_name, max_lag: int = 20,
                                    z: float = 1.0) -> pd.DataFrame:
    """
    based on https://www.analyticsvidhya.com/blog/2021/08/multivariate-time-series-anomaly-detection-using-var-model/

    :param z:
    :param df:
    :param time_column_name:
    :param max_lag:
    :return:
    """
    time_df = df.set_index(time_column_name)

    var = VAR(df)

    # select the best lag order
    lag = var.select_order(max_lag).aic

    var_fitresults = var.fit(lag)

    #  The squared errors are then used to find the threshold, above which the
    #  observations are considered to be anomalies.
    squared_errors = var_fitresults.resid.sum(axis=1) ** 2

    # We can also use another method to find thresholds like finding the 90th percentile of the
    # squared errors as the threshold. The best value for z is considered to be between 1 and 10.
    threshold = np.mean(squared_errors) + (z * np.std(squared_errors))
    anomalies = (squared_errors >= threshold).astype(int)

    time_df = time_df.reset_index().iloc[lag:, :]
    time_df['anomaly'] = anomalies.values
    time_df['squared_errors'] = squared_errors
    return time_df


data = vector_autoregression_anomalies(df, "time_s", z=.2, max_lag=20)

data.to_csv("out.csv")
print(data)


fig, ax = plt.subplots(figsize=(17, 10))

for (t, p) in zip(data.time_s, data.anomaly):
    if p == 1:  # outlier
        ax.axvline(x=t, color="lightgray")

ax.plot(
    data.time_s, data.lin_acc_x,
    # data.time_s, data.lin_acc_y,
    # data.time_s, data.lin_acc_z,
    data.time_s, data.lin_acc_abs
)
ax.set_xlabel('time')
ax.set_ylabel('acc m/s^2')

fig.savefig("out.png")
