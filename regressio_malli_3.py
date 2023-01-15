import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
data = np.array([[174, 94], [189, 87], [185, 102],
                 [195, 104], [149, 61], [174, 91],
                 [161, 81], [185, 79], [161, 65],
                 [140, 47], [176, 54], [172, 85],
                 [187, 89], [164, 70], [143, 56],
                 [191, 93], [172, 93], [168, 59]])
df = pd.DataFrame(data, columns=["pituus", "paino"])
pituus =  df.pituus
paino = df[['paino']]
paino1 = paino.to_numpy()

pituus2 = np.dot(paino, np.array([1, 2])) + 3

testipainot = pd.DataFrame({"paino": [65, 75, 85]})
#==================================================
reg = LinearRegression().fit(paino2, pituus2,sample_weight=np.array([161.96, 169.07, 176.19]))
reg.score(paino2, pituus2, sample_weight=np.array([161.96, 169.07, 176.19]))
reg.coef_
reg.intercept_
ennusteet = reg.predict(pituus2)
print(ennusteet)
#==================================================
