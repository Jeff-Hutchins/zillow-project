from pydataset import data
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from env import user, host, password
import wrangle
import split_scale
from statsmodels.formula.api import ols
from math import sqrt


def plot_residuals(X, y, dataframe):
    return sns.residplot(X, y, dataframe)


def regression_errors(y, yhat):
    SSE = sum(df['residual-2'])
    ESS = sum((df.yhat - df.y.mean())**2)
    TSS = ESS + SSE
    MSE = SSE/len(df)
    RMSE = sqrt(MSE)
    df_eval = pd.DataFrame(np.array(['SSE', 'ESS', 'TSS', 'MSE', 'RMSE',]), columns=['metric'])
    df_eval['model_error'] = np.array([SSE, ESS, TSS, MSE, RMSE])
    return df_eval


def baseline_mean_errors(y):
    yhat_baseline = y.mean()
    yhat_baseline = y.median()
    SSE_baseline = sum(df['residual-2'])
    MSE_baseline = SSE/len(df)
    RMSE_baseline = sqrt(MSE)
    return SSE, MSE, RMSE


def better_than_baseline(SSE, SEE_baseline):
    return SSE < SSE_baseline


def model_significance(ols_model):
    R2 = round(ESS/TSS *100, 2)
    print(f'{R2} percent of the variance is explained by this model.')
    regr_results = regr.summary()
    regr_pvalues = pd.DataFrame(regr.pvalues)
    regr_x_pvalue = regr_pvalues.loc['x', 0]
    if regr_x_pvalue < .005:
        return f'correlation between the model and the tip values are statistically signifigant'
    else:
        return f'correlation between the model and the tip values are not statistically signifigant'