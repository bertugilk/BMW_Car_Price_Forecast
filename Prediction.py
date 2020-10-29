import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow_core.python.keras.models import load_model

data=pd.read_excel("bmw.xlsx")

#----------------- Understanding The Data -------------------:

#print(data.head())
#print(data.isnull().sum())

#sbn.distplot(data["price"])
#sbn.countplot(data["year"])
#plt.show()

highestPricedCars=data.sort_values("price",ascending=False).head(20)
#print(highestPricedCars)

#--------------------- Data Cleaning -------------------------:

#print(len(data)*0.01)
newData=data.sort_values("price",ascending=False).iloc[107: ]
#print(newData.groupby("year").mean()["price"])

data=newData

data=data.drop(["model","transmission","fuelType"],axis=1) # We extract the string values from the data.
#print(data)

#--------------------- Create Model -------------------------:

y=data["price"].values
X=data.drop("price",axis=1).values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

model=Sequential()

model.add(Dense(18,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),batch_size=250,epochs=300)
model.save("BMW_Model.h5")

#----------------------------- Results ----------------------------:

predictArray=model.predict(X_test)
#print(predictArray)
loss=mean_absolute_error(y_test,predictArray)
#print(loss)

plt.scatter(y_test,predictArray)
plt.plot(y_test,y_test,"g-*")
#plt.show()

year=int(input("Year: "))
mileage=float(input("Mileage: "))
tax=float(input("Tax: "))
mpg=float(input("MPG: "))
engineSize=float(input("Engine Size: "))

print("\n")

newValues=[[year,mileage,tax,mpg,engineSize]]
newValues=scaler.transform(newValues)

newPredict=model.predict(newValues)
print("Predict: ",int(newPredict))