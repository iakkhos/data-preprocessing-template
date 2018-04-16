# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 18:47:53 2018

@author: ozano
"""

#1. libraries

import pandas as pd #veri önişleme ve veri üzerine kullanılan kütüphane. Dağınık yapılı değil, makinenin kapasitesine göre sınırlıdır, büyük veriler için uygun değil.
import numpy as np 
import matplotlib.pyplot as plt

#2. ************DATA PREPROCESSING************

#2.1 ***Data Loading***
veriler = pd.read_csv("satislar.csv")


aylar = veriler[['Aylar']]

satislar = veriler[['Satislar']]


#2.5 ***Splitting data test and train sets***

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)


#2.6 ***Scaling datas and standartization***(öznitelik ölçekleme)


'''
from sklearn.preprocessing import StandardScaler #standart sapma

sc = StandardScaler() #obje oluşturduk

x_train = sc.fit_transform(x_train) # fit = makineye öğret, transform=uygula
x_test = sc.fit_transform(x_test)
'''

# Model Constructing ( Linear Regression )

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train) #öğrenme aşaması, x_train bağımsız veriler ile y_train bağımlı değişkenini nasıl tahmin edeceğini öğreniyor, linear regression algoritması ile

tahmin = lr.predict(x_test) #x_test'teki bağımsız veriler ile bir satislar tahmini yapar


# Plotting

x_train = x_train.sort_index() #bir pandas fonksiyonu, index numaralarına göre dataframe verilerini sıralar
y_train = y_train.sort_index()
plt.plot(x_train,y_train)

plt.plot(x_test,tahmin)
plt.title("aylık kazanç")
plt.xlabel("ay")
plt.ylabel("kazanc")