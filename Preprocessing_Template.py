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
veriler = pd.read_csv("eksikveriler.csv")

#2.2 ***Missing Values***

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)

yas = veriler.iloc[:,1:4].values

#imputer = imputer.fit(yas[:,1:4]) #eğitim
#yas[:,1:4] = imputer.transform(yas[:,1:4]) #uygulama

yas[:,1:4] = imputer.fit_transform(yas[:,1:4])

print(yas)



#2.3 ***Enoder: nominal to numeric***


ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder #labeldaki nominal verilere sayısal değerler atar.

le = LabelEncoder()

ulke[:,0] = le.fit_transform(ulke[:,0])

print(ulke)


from sklearn.preprocessing import OneHotEncoder # sayısal değerler atanmış nomimal verileri, sütunlaştırarak kategorize eder

ohe = OneHotEncoder(categorical_features="all")

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)



#2.4 ***From numpy to data frames***


#verileri dataframe'den ayırdığımızda NumPy array ya da ndarray gibi farklı tiplerde ayrıldılar. Bunları geri dataframe
#formuna sokup, daha sonra geri birleştirmeliyiz...
sonuc = pd.DataFrame(data = ulke, index=range(22),columns=['fr','tr','us'])
print(sonuc)

sonuc2=pd.DataFrame(data = yas , index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

sonuc3=pd.DataFrame(data = cinsiyet, index=range(22),columns=['cinsiyet'])
print(sonuc3)


#2.5 ***Dataframe concentanetion***

s = pd.concat([sonuc,sonuc2],axis=1) #dataframleri birbiriyle birleştirir (axis satırdan mı sütündan mı birleştirileceğine karar verir)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)


#2.5 ***Splitting data test and train sets***

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)


#2.6 ***Scaling datas and standartization***(öznitelik ölçekleme)


#veri başlıkları, birbirinden çok farklı dünyalarda olabilirler. Mesela boy ile, kilo verileri ikisi de sayısal olsa da,
#değerlerini alış biçimleri bakımından birbirinden farklıdır ve beraber kullanmakta sakıncalar olabilir. Bu sebeple,
#normalization ya da standart sapma yöntemleriyle veri sütunumuzu belirli bir değer aralığına indirgeyebiliriz.

from sklearn.preprocessing import StandardScaler #standart sapma

sc = StandardScaler() #obje oluşturduk

x_train = sc.fit_transform(x_train) # fit = makineye öğret, transform=uygula
x_test = sc.fit_transform(x_test)

