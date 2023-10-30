import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# Define the custom Perceptron class
class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, max_iterations=280):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for i in range(self.max_iterations):
            for j, x in enumerate(X):
                y_pred = self.predict(x)
                if y[j] * y_pred <= 0:
                    self.weights += self.learning_rate * y[j] * x
                    self.bias += self.learning_rate * y[j]
        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

#LeastSquares
class OLS(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        # Add a column of ones to X to include the intercept term in the model
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate the coefficients using the closed-form solution
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

    def predict(self, X):
        # Add a column of ones to X to include the intercept term in the model
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate the predicted values using the learned coefficients
        y_pred = X @ self.coef_

        return y_pred


df = pd.read_csv('housing.csv')

# Numerical features
numerical_features = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]

# Categorical features
categorical_features = df[['ocean_proximity']]

# Scale numerical features
numerical_transformer = MinMaxScaler()
numerical_features_scaled = numerical_transformer.fit_transform(numerical_features)

# Apply one hot encoding to the categorical feature
categorical_transformer = OneHotEncoder()
categorical_features_encoded = categorical_transformer.fit_transform(categorical_features)

# Fill the missing values with the median value of each feature
numerical_features = numerical_features.fillna(numerical_features.median())

# Save the filled numerical features back to the DataFrame
df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']] = numerical_features

# Combine the numerical and categorical features
X = np.hstack((numerical_features_scaled, categorical_features_encoded.toarray()))

# Fill the missing values with the median value of each feature
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Replace categorical feature with numerical values
df['ocean_proximity'] = df['ocean_proximity'].replace({'<1H OCEAN': 0, 'INLAND': 0.25, 'ISLAND': 0.5,'NEAR BAY': 0.75, 'NEAR OCEAN':1 })
y = df['median_house_value'].values


for column in df.columns:
    plt.hist(df[column].dropna(), bins=10, alpha=0.5, label=column)
    plt.legend()
    plt.show()


# Scatter matrix plot between multiple variables
sns.pairplot(df[['longitude', 'latitude', 'total_rooms']])
plt.show()


###Perceptron code

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_list = []
mae_list = []
for train_index, test_index in kf.split(X_train):
    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]

    model = Perceptron(learning_rate=0.01, max_iterations=280)
    model.fit(X_cv_train, y_cv_train)
    y_pred = model.predict(X_cv_test)

    mse_list.append(mean_squared_error(y_cv_test, y_pred))
    mae_list.append(mean_absolute_error(y_cv_test, y_pred))

print("Mean Squared Error: ", np.mean(mse_list))
print("Mean Absolute Error: ", np.mean(mae_list))

# Train on the full training set and evaluate on the test set
model = Perceptron(learning_rate=0.01, max_iterations=280)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Test Mean Squared Error: ", mse)
print("Test Mean Absolute Error: ", mae)

# Use 10-fold cross validation to calculate the accuracy of the classifier
k = 10
scores = cross_val_score(Perceptron(learning_rate=0.01, max_iterations=280), X, y, cv=k)

# Evaluate the model on the test set
test_accuracy = model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)


###Least squares
X = df.drop('median_house_value', axis=1).values
y = df['median_house_value'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the model on the training set and evaluate on the test set
model = OLS()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Test Mean Squared Error: ", mse)
print("Test Mean Absolute Error: ", mae)



#Multi-layer neural network

#defining the neural network model
model = Sequential()
n_samples, n_features = df.shape
l=n_features
model.add(Dense(10, input_dim=l, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

#Compliling the model
model.compile(loss='mean_squared_error', optimizer='adam')

#making a dataset example
X, y = make_regression(n_samples=1000, n_features=l, noise=0.1)

# Using 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

# Printing the results of the mse and mae
print("MSE: %.2f" % results.mean())
print("MAE: %.2f" % abs(results).mean())