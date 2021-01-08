"""
Standard Normal Distribution
Author: Balamurali M
https://medium.com/@balamurali_m/normal-distribution-with-python-793c7b425ef0
"""

import numpy as np
import matplotlib.pyplot as plt


class norm1:
    def __init__(self, a1, b1, c1):
        self.a1 = a1    #Mean Value
        self.b1 = b1    #Standard deviation
        self.c1 = c1    #The X value
        
        
    def dist_curve(self):
        plt.plot(self.c1, 1/(self.b1 * np.sqrt(2 * np.pi)) *
            np.exp( - (self.c1 - self.a1)**2 / (2 * self.b1**2) ), linewidth=2, color='y')
        plt.show()

#mean 0 and sd 1 for the standard normal distribution
mean = 0
sd = 1


c = np.random.normal(mean, sd, 3000)
        
w1, x1, z1 = plt.hist(c, 100, normed=True) #hist

hist1 = norm1(mean, sd, x1)
plot1 = hist1.dist_curve()


#Linear Regression using Statsmodel
import numpy as np
import statsmodels.api as sm
spector_data = sm.datasets.spector.load(as_pandas=False)   
spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)     #exogenous: caused by factors outside the system
mod = sm.OLS(spector_data.endog, spector_data.exog)#Fitting OLS
res = mod.fit()
print(res.summary())





