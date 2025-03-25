import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


X = np.array([[2000, 3, 1], [1500, 2, 2], [1800, 3, 3], [1200, 2, 1], [2200, 4, 2]])
y = np.array([500000, 350000, 450000, 300000, 550000])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")