import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\elcot\\Downloads\\Position_Salaries.csv")

#sns.heatmap(data.corr(),annot=True)  0.82 relationship

x=data.iloc[:,[1]].values

y=data.iloc[:,[2]].values


'''plt.scatter(x,y)
plt.show()'''


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures()
x_train=poly.fit_transform(x_train)
x_test=poly.fit_transform(x_test)
reg=LinearRegression()
reg.fit(x_train,y_train)
#polynomial prediction
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

