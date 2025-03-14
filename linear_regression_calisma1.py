import pandas as pd

df1=pd.read_csv("et_fiyat.csv",sep=";")

#%%LINEAR REGRESSION
#ADIMLAR:
    #1.X VE Y EKSENİNİ BELİRLE YANİ GERÇEK NOKTALARI
    #2.BU X VE Y EKSENİNİ KULLANARAK BİR LİNE FİT UYDUR
    #3.B0,B1'İ BULARAK DENKLEMİ OLUŞTUR
    #4.OLUŞAN DENKLEME KARŞILIK GELEN X EKSENİ TAHMİNLERİNİ BUL
    #5.LİNE VE GERÇEK DEĞERLERİNİN GRFİĞİNİ ÇİZDİR
    
#%%
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

linear=LinearRegression()
#1.adım
xekseni=df1["etmiktari"].values.reshape(-1,1)
yekseni=df1["fiyat"].values.reshape(-1,1)

plt.scatter(xekseni,yekseni)
plt.show()

#2.adım
linear.fit(xekseni,yekseni)

#3.adım
b0=linear.intercept_
print("b0:",b0)
b1=linear.coef_
print("b1:",b1)
#fiyat=b0+b1*etmiktari
fiyat=b0+b1*510
print("510 gram et fiyati:",fiyat)

#4.adım
tahmin=linear.predict(xekseni)
print("tahmini degerler:",tahmin)
plt.plot(xekseni,tahmin,label="LINEAR REGRESSIN",color="black")
plt.xlabel("etmiktari(gr)")
plt.ylabel("fiyat")
plt.legend()
plt.show()

rkare=r2_score(yekseni,tahmin)
print("R^2:",rkare)
mse = mean_squared_error(yekseni, tahmin)
print("MSE:", mse)





