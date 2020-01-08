#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#%%
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#%%
#fits simple linear regression to the training set
model = LinearRegression()
model.fit(X_train, y_train)

#%%
#predicts outcome using the model 
y_pred = model.predict(X_test)

#%%
#training set results
plt.scatter(X_train, y_train, color = 'magenta')
plt.plot(X_train, model.predict(X_train), color = 'slategrey')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#%%
#test set results
plt.scatter(X_test, y_test, color = 'magenta')
plt.plot(X_train, model.predict(X_train), color = 'slategrey')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
