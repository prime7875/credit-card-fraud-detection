# credit card fraud detection

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv("D:/csv files/creditcard.csv")
print(credit_card_data.head())
print("\n")
print(credit_card_data.tail())
print("\n")

## getting some info from our csv file
print(credit_card_data.info())


print(credit_card_data.isnull().sum())
## checking the normal transction and fraudulant transaction
## 0 = normal nd 1 = fraud
print(credit_card_data['Class'].value_counts())

## seperating the data 
normal = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
print()
print("data for normal trans")
print(normal.Amount.describe())
print("\n")
print("data for fraud trans")
print(fraud.Amount.describe())

### compare the data
print(credit_card_data.groupby("Class").mean())

normal_sample = normal.sample(n=492)

## adding normal_sample and fraud

new_data = pd.concat([normal_sample,fraud],axis = 0)

X = new_data.drop(columns="Class", axis=1)
Y = new_data['Class']

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

model = LogisticRegression()

model.fit(x_train,y_train)

### accuracy for x_train data
x_train_prrediction = model.predict(x_train)
x_train_accuracy = accuracy_score(x_train_prrediction,y_train)
print(x_train_accuracy)
print(x_train_prrediction)

### accuracy for x_test data
x_test_prediction = model.predict(x_test)
x_test_accuracy = accuracy_score(x_test_prediction,y_test)
print(x_test_accuracy)
