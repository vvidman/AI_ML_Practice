# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

## LOAD and PREPARE data

# Sample dataset (house prices based on square footage)
data = {
    'SquareFootage': [500, 1000, 1500, 1800, 2400, 3000, 3500, 4000, 4500],
    'Price': [50000, 100000, 200000, 250000, 300000, 350000, 400000, 500000, 600000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())

## Splitting the data into training and testing sets

# Features (X) and Target (y)
X = df[['SquareFootage']]  # Feature(s)
y = df['Price']            # Target variable

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and testing sets
#print(f"Training data: {X_train.shape}, {y_train.shape}")
#print(f"Testing data: {X_test.shape}, {y_test.shape}")

## Training the linear regression model

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Display the learned coefficients
#print(f"Intercept: {model.intercept_}")
#print(f"Coefficient: {model.coef_[0]}")

'''
The intercept and coefficient are the parameters that define the linear equation y=mx+b, where:
 m is the coefficient (slope), and
 b is the intercept (y-axis intercept).
 '''

## Making predictions

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Display the predictions
#print('Prediction based on: ', X_test.values)
#print("Predicted Prices:", y_pred)
#print("Actual Prices:", y_test.values)

## Evaluating the model

'''
It’s important to evaluate the model to see how well it performed on the test data. 
We’ll use mean squared error (MSE) and R-squared (R²) as performance metrics:
'''
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
#print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
model_fittness = bool(1.0 - abs(r2) < 0.02)

print('The model fitness falls within the acceptance criteria:', model_fittness)

'''
MSE gives the average squared difference between the actual and predicted values (the lower, the better).
R² tells you how well the model fits the data (1 means a perfect fit, while 0 indicates no fit).
'''

## Visualizing the results

# Plot the data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('House Prices vs. Square Footage')
plt.legend()

# Show the plot
plt.show()