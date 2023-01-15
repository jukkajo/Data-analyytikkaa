import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X = X.drop(['Latitude', 'Longitude'], axis=1) # Jätetään pois latitude ja longitude muuttujat
X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=1) # valitaan 10% havainnoista

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

import scipy.stats as stats

zscores1 = stats.zscore(X_train)
zscores2 = stats.zscore(y_train)
zscores1_2 = stats.zscore(y_test)
zscores2_2 = stats.zscore(X_test)

X_train = zscores1
y_train = zscores2 
y_test =zscores1_2
X_test = zscores2_2

from sklearn.neighbors import KNeighborsRegressor
kn_reg = KNeighborsRegressor(n_neighbors=2).fit(zscores1, zscores2)
model = kn_reg

from sklearn.metrics import mean_squared_error
MSE_train = mean_squared_error(y_train, model.predict(X_train))
MSE_test = mean_squared_error(y_test, model.predict(X_test))

print("Train MSE:", MSE_train)
print("Test MSE:", MSE_test)

