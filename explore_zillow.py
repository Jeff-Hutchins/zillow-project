import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from env import user, host, password
import wrangle
import split_scale



def plot_variable_pairs(df):
    g=sns.PairGrid(df)
    g.map(sns.regplot)
    plt.show()


def plot_categorical_and_continuous_vars(df):
    plt.figure(figsize=(16,8))
    plt.subplot(1,3,1)
    plt.bar(df.tenure_years,df.total_charges)
    plt.xlabel('Tenure in years')
    plt.ylabel('Total charges in dollars')
    plt.subplot(1,3,2)
    sns.stripplot(df.tenure_years,df.total_charges)
    plt.subplot(1,3,3)
    plt.scatter(df.tenure_years,df.total_charges)
    plt.pie(df.groupby('tenure_years')['total_charges'].sum(),labels=list(df.tenure_years.unique()),autopct='%1.1f%%',shadow=True)
    plt.title(" Percent of total charges by tenure")
    plt.show()