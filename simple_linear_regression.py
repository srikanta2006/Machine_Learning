#Dataset: Tips
#model: simple linear regression

import seaborn as sns # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore

#1
# Load the example Tips dataset
df = sns.load_dataset('tips')
df.head()

#2
# Display basic information about the dataset
df.info()

# Display summary statistics
df.describe()

#3
# Scatter plot of total_bill vs tip
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='total_bill', y='tip')
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()

# Observation:
# when bill increases, tip also increases.
# relationship appears to be straight line.
# This suggests that a simple linear regression model may be appropriate.

#4
#feature and target separation
X = df[['total_bill']]  # feature
y = df['tip']           # target

#5
# Split the dataset into training and testing sets
# a model must perform well on unseen data.
from sklearn.model_selection import train_test_split # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 80% training and 20% testing
#feature standardization why?
    #compare the coefficients fairly
    #avoid the dominance of large scale features
    #prepare for ridged and lasso regression

#6
from sklearn.preprocessing import StandardScaler # type: ignore
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# fit on training data and transform both training and testing data?
# because we want to avoid data leakage from test set to training set.

#7
# Train the simple linear regression model
from sklearn.linear_model import LinearRegression # type: ignore
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("coefficient(m):", model.coef_[0])
print("intercept(c):", model.intercept_)
# y = mx + c
# y = tip, x = total_bill
# tip = m*total_bill + c
# tip = model.coef_[0] * total_bill + model.intercept_
# coefficient(m) indicates the change in tip for a one-unit change in total_bill.
# intercept(c) indicates the tip when total_bill is zero.


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

#8
from sklearn.metrics import r2_score # type: ignore = r2_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("R² score:", r2)

from sklearn.metrics import mean_squared_error # type: ignore
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
# R² score indicates the proportion of variance in the dependent variable that is predictable from the independent variable.
# Mean Squared Error (MSE) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(scaler.transform(X)), color='red', label='Regression Line')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Simple Linear Regression: Total Bill vs Tip')
plt.show()
# The scatter plot shows the actual data points (in blue) and the regression line (in

#Input: total_bill, Output: tip)

bill_amount = float(input("Enter the total bill amount: "))
bill_scaled = scaler.transform([[bill_amount]])
tip_prediction = model.predict(bill_scaled)
print(f"Predicted tip for a bill of ${bill_amount}: ${tip_prediction[0]:.2f}")
# This visualization helps to understand how well the regression line fits the data.

