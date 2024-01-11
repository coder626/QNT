import pandas as pd
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('HSDcopy.csv')

print(df.iloc[0])


#df = df.drop(df.index[0])
df = df.iloc[1:, 1:]

print(df.iloc[0])

def convert_percent_to_decimal(value):
    if isinstance(value, str) and value.endswith('%'):
        return float(value.strip('%')) / 100
    else:
        return value

df = df.applymap(convert_percent_to_decimal)


"""
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        if df.iloc[i,j] == 5:  # check if the current element is equal to 5
            df.iloc[i,j] = 10  # modify the element to be equal to 10
"""

print(df.iloc[0])


def replace_num(x):
    x = float(x)
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

df = df.applymap(replace_num)



print(df.iloc[0])

print(df.shape)

data = pd.DataFrame.transpose(df)

print(data.shape)

first_column = data.iloc[:, 0]

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=26, batch_size=16)

labels = model.predict(X_test)

print(X_test.shape)

mse = model.evaluate(X_test, y_test)
print(mse)

print(labels)

print(df.iloc[:, 0])

print(y_test)

print(labels.shape)




