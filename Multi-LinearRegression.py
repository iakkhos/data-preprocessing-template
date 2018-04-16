# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:46:22 2018

@author: Ozan Yurtsever

"""

#1. libraries

import pandas as pd #veri önişleme ve veri üzerine kullanılan kütüphane. Dağınık yapılı değil, makinenin kapasitesine göre sınırlıdır, büyük veriler için uygun değil.
import numpy as np 
import matplotlib.pyplot as plt


#2. ************DATA PREPROCESSING************

#2.1 ***Data Loading***
veriler = pd.read_csv("veriler.csv")




#2.3 ***Encoder: nominal to numeric***


ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder #labeldaki nominal verilere sayısal değerler atar.

le = LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke)


from sklearn.preprocessing import OneHotEncoder # sayısal değerler atanmış nomimal verileri, sütunlaştırarak kategorize eder

ohe = OneHotEncoder(categorical_features="all")

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)



c = veriler.iloc[:,-1:].values

c[:,0] = le.fit_transform(c[:,0])

print(c)


#2.4 ***From numpy to data frames***


#verileri dataframe'den ayırdığımızda NumPy array ya da ndarray gibi farklı tiplerde ayrıldılar. Bunları geri dataframe
#formuna sokup, daha sonra geri birleştirmeliyiz...
sonuc = pd.DataFrame(data = ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)



cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

sonuc2=pd.DataFrame(data = c, index=range(22),columns=['cinsiyet'])
print(sonuc2)


#2.5 ***Dataframe concentanetion***


s = pd.concat([sonuc,veriler.iloc[:,1:4]],axis=1)

print(s)


#2.6 ***Splitting data test and train sets***

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc2,test_size=0.33,random_state=0)


# Model Constructing ( Linear Regression )

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train) #öğrenme aşaması, x_train bağımsız veriler ile y_train bağımlı değişkenini nasıl tahmin edeceğini öğreniyor, linear regression algoritması ile

tahmin = lr.predict(x_test) #x_test'teki bağımsız veriler ile bir satislar tahmini yapar

sol = s.iloc[:,:3]
sag = s.iloc[:,4:]
boy=  s.iloc[:,3:4].values
veri = pd.concat([sol,sag],axis=1)

veri = pd.concat([veri,sonuc2],axis=1)


# Backward Elimination

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values=veri,axis = 1) #formül için bir sabit sayı ekliyoruz
X = pd.DataFrame(data=X,index=range(22),columns=['sabit','fr','tr','us','kilo','yas','cinsiyet'])

x_l = veri.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = boy , exog = x_l).fit()
print(r_ols.summary())



