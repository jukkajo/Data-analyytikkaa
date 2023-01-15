import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
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

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
tarkkuus = logisticRegr.score(X_test, y_test)

