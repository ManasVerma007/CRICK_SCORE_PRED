from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import sklearn as skl
import pandas as pd
import numpy as np

dataset = pd.read_csv('PREPROCCESED40.csv')
x = dataset[['boundry_runs', 'wickets fallen', 'run_rate']]
y = dataset[['total_score']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.01, random_state=0)

lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)

y_pred = lasso.predict(x_test)
print(y_pred)

f = mean_squared_error(y_test, y_pred)
print(f)