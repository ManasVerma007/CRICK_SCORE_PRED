from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn as skl
import pandas as pd
import numpy as np


dataset = pd.read_csv('PREPROCCESED40.csv')
x = dataset[['boundry_runs', 'wickets fallen', 'run_rate']]
y = dataset[['total_score']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=100)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(y_pred)

k = float(mean_squared_error(y_test, y_pred))
print(k)