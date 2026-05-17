from scipy.stats import ttest_ind

import numpy as np 

n_1=np.array([2.8,3.1,2.9,3.4,3.0,2.7,3.2],dtype=np.float32)
n_2=np.array([2.4,2.2,2.5,2.6,2.3,2.4,2.7,2.5,2.1,2.6],dtype=np.float32)
stats,p=ttest_ind(n_1,n_2)
print(f"stats: {stats}\np:{p}")
print( "la hipotesis nula se rechaza" if p<0.05 else "La hipotesis nula se acepta")
