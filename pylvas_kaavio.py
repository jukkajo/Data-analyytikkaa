import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=137)
x = pd.Series(rng.integers(low=1, high=7, size=20))
#====================Data===========================
arvot, lkmt = np.unique(x, return_counts=True)
#dictionary luvuista ja esiitymista
data = {}
for ind in range(len(arvot)):
    data[str(arvot[ind])] = lkmt[ind]
a = list(data.keys())
l = list(data.values())
f = plt.figure(figsize = (10, 5))
#====================Graph===========================
plt.bar(l, a, color ='green', width = 0.8)
plt.xlabel("Nopan silmäluvut")
plt.ylabel("Frekvenssit")
plt.title("Nopanheitot")

#print(arvot, lkmt)
#plt.show()
plt.savefig("image.png")
