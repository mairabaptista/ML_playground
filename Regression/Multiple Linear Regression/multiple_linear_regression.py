# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# %%
# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
print(X)
onehotencoder = ColumnTransformer(
    transformers=[
        ("City",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]            # The column(s) to be applied on.
         )
    ], remainder='passthrough'
)
X = onehotencoder.fit_transform(X)

# %%
"""labelencoder = LabelEncoder()
y[:] = labelencoder.fit_transform(y[:])
onehotencoder = ColumnTransformer(
    transformers=[
        ("City",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]            # The column(s) to be applied on.
         )
    ], remainder='passthrough'
)
y = np.reshape(y, (50, 1))
y = onehotencoder.fit_transform(y)"""

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# %%
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %%
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# %%
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""