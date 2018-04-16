# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:41:27 2018

@author: ozano
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("maaslar.csv")

# DataFrame Slicing

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

# linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

# polynomial regression

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=6)

x_poly = pr.fit_transform(x) # elimizdeki veriyi polinomal hale getirir ( x^0 + x^1 + x^2 + ..... x^n) 
lr2 = LinearRegression()
lr2.fit(x_poly,y)



#visualization


plt.scatter(x,y,color='red')
plt.plot(x,lr.predict(x),color='blue')
plt.show()


plt.scatter(x,y)
plt.plot(x,lr2.predict(x_poly))
plt.show()
