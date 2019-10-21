from sklearn.linear_model import LinearRegression
from sklearn import metrics

def modeling_function(X_train,X_test,y_train,y_test):
    #model 1
    lm1=LinearRegression()
    lm1.fit(X_train_scaled[['bedrooms', 'bathrooms', 'square_feet', 'lot_size_minus_sqft']],y_train)
    lm1_predictions=lm1.predict(X_train_scaled[['bedrooms', 'bathrooms', 'square_feet', 'lot_size_minus_sqft']])
    predictions['lm1']=lm1_predictions
    
    #baseline model
    predictions['baseline'] = y_train.mean()[0]
    predictions.head()
    
    return predictions


def plot_regression():
    pd.DataFrame({'actual': y_train,
              'lm1': y_pred_lm1.ravel(),
              'lm_baseline': y_pred_baseline.ravel()})\
    .melt(id_vars=['actual'], var_name='model', value_name='prediction')\
    .pipe((sns.relplot, 'data'), x='actual', y='prediction', hue='model')
    min = 0
    max = 5_000_000
    plt.plot([min, max],[min, max], c='black', ls=':')