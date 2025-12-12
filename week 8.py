from pandas import read_csv

header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
my_data = read_csv("pima_indians_diabetes.csv", names=header_names)

print(my_data.shape)
print("Printing the first 5 data")
print(my_data.head(5))
print("\n------\n")
print("Printing the last 5 data")
print(my_data.tail(5))

from pandas import read_csv

header_names = [
    'preg', 'plas', 'pres', 'skin',
    'test', 'mass', 'pedi', 'age', 'class'
]

my_data = read_csv("pima_indians_diabetes.csv", names=header_names)

print(my_data.shape)
print(my_data.dtypes)

from pandas import read_csv

header_names = ['preg', 'plas', 'pres', 'skin', 'test'
    , 'mass', 'pedi', 'age', 'class']
my_data = read_csv("pima_indians_diabetes.csv", names=header_names)
print(my_data.shape)
count_diabetics_class = my_data.groupby('class').size()
print(count_diabetics_class)

from pandas import read_csv
import matplotlib.pyplot as plt  # Correct import

header_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

# Must match number of headers (9 values → add one more)
header_count = [32, 43, 56, 6, 75, 12, 43, 41, 54]

# Read CSV
my_data = read_csv("pima_indians_diabetes.csv", names=header_names)

# Create the figure
plt_fig = plt.figure()

# Correct axes: values must be between 0 and 1
plt_ax = plt_fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Create bar chart
plt_ax.bar(header_names, header_count)

plt_ax.set_title("Header Count Bar Chart")
plt_ax.set_xlabel("Attributes")
plt_ax.set_ylabel("Count")

plt.show()

#visualising our data
# visualised our data as Piechart
from pandas import read_csv
from matplotlib import pyplot  #import matplotlib

header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
header_count = [32, 43, 56, 6, 75, 12, 43, 41, 54]
plt_fig = pyplot.figure()
plt_ax = plt_fig.add_axes([0, 0, 1, 1])
plt_ax.pie(header_count, labels=header_names, autopct='&1.2f%%')
pyplot.show()

from pandas import read_csv
from matplotlib import pyplot
import pandas as pd

header_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

# Load CSV
my_data = read_csv("pima_indians_diabetes.csv", names=header_names, header=None)

my_data = my_data.apply(pd.to_numeric, errors='coerce')

# Plot boxplots
my_data.plot(kind='box', subplots=True, layout=(3, 3),
             sharex=False, sharey=False, figsize=(12, 8))

pyplot.show()

import pandas as pd
from pandas import read_csv
from matplotlib import pyplot

header_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

# Load CSV
my_data = read_csv("pima_indians_diabetes.csv", names=header_names, header=None)


my_data = my_data.apply(pd.to_numeric, errors='coerce')

# Density plots
my_data.plot(
    kind='density',
    subplots=True,
    layout=(3, 3),
    sharex=False,
    figsize=(12, 8)
)

pyplot.show()


from pandas import read_csv

header_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

my_data = read_csv("pima_indians_diabetes.csv",
                   names=header_names,
                   header=None)

print(my_data.skew(numeric_only=True))

from pandas import read_csv
from matplotlib import pyplot
import pandas as pd

# FIXED: comma after 'pres'
header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# FIX: your CSV already has headers → remove names=header_names
my_data = read_csv("pima_indians_diabetes.csv")

my_data.hist(bins=10, figsize=(10, 8))
pyplot.show()

from pandas import read_csv

header_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

my_data = read_csv("pima_indians_diabetes.csv",
                   names=header_names,
                   header=None)

# Convert all columns to numeric (important if CSV contains text)
my_data = my_data.apply(pd.to_numeric, errors='coerce')

mydata_correlations = my_data.corr(method='pearson')
print(mydata_correlations)

import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt

# -----------------------------
# Column names (only needed if CSV has no headers)
header_names = ['preg', 'plas', 'pres', 'skin', 'test',
                'mass', 'pedi', 'age', 'class']

# -----------------------------
# Load CSV
# If your CSV already has headers, remove 'names=header_names' and 'header=None'
my_data = read_csv("pima_indians_diabetes.csv",
                   names=header_names,
                   header=None)

# Convert all columns to numeric
my_data = my_data.apply(pd.to_numeric, errors='coerce')

# -----------------------------
# Compute correlation matrix
data_correlations = my_data.corr()

# -----------------------------
# Plot the correlation matrix
corr_fig = plt.figure()
ax = corr_fig.add_subplot(111)

cax = ax.matshow(data_correlations, vmin=-1, vmax=1, cmap='coolwarm')
corr_fig.colorbar(cax)

# Set ticks and labels
ticks = np.arange(len(header_names))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(header_names, rotation=90)
ax.set_yticklabels(header_names)

plt.title("Correlation Matrix of Pima Indians Diabetes Dataset", pad=20)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------
# Generate random dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = 3 + 5 * x + np.random.rand(100, 1)  # add small noise to make it realistic

# -----------------------------
# Initialize Linear Regression model
lr_regress = LinearRegression()

# Fit the regression model
lr_regress.fit(x, y)

# Predict
y_pred = lr_regress.predict(x)

# -----------------------------
# Evaluate the model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# -----------------------------
# Print results
print('Slope:', lr_regress.coef_)
print('Intercept:', lr_regress.intercept_)
print('Mean squared error:', mse)
print('Mean absolute error:', mae)
print('Root mean squared error:', rmse)
print('R2 score:', r2)

# -----------------------------
# Plot data points and regression line
plt.scatter(x, y, s=10, label='Data points')
plt.plot(x, y_pred, color='r', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

# -----------------------------
# Predict a new value
pred_my_value = lr_regress.predict([[0.5]])  # Example: x=0.5
print("Prediction for x=0.5:", pred_my_value)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Load dataset
# If your CSV already has headers, remove 'names=header_names' and 'header=None'
header_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
my_data = pd.read_csv("pima_indians_diabetes.csv", names=header_names, header=None)

# -----------------------------
# Convert feature and target to numeric
x = pd.to_numeric(my_data['preg'], errors='coerce').values.reshape(-1, 1)  # Feature: preg
y = pd.to_numeric(my_data['age'], errors='coerce').values.reshape(-1, 1)   # Target: age

# -----------------------------
# Remove rows with NaN values
valid_rows = ~np.isnan(x).flatten() & ~np.isnan(y).flatten()
x = x[valid_rows]
y = y[valid_rows]

# -----------------------------
# Train/Test split (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1
)

# -----------------------------
# Initialize and train Linear Regression model
lr_regress = LinearRegression()
lr_regress.fit(x_train, y_train)

# Predict
y_pred = lr_regress.predict(x_test)

# -----------------------------
# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Slope:", lr_regress.coef_)
print("Intercept:", lr_regress.intercept_)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# -----------------------------
# Plot actual vs predicted
plt.scatter(x_test, y_test, s=10, label="Actual")
plt.plot(x_test, y_pred, color='r', label="Predicted")
plt.xlabel("Pregnancy (preg)")
plt.ylabel("Age")
plt.title("Simple Linear Regression: Age vs Pregnancy")
plt.legend()
plt.show()

# -----------------------------
# Predict for a specific value
pred_value = lr_regress.predict([[4]])  # Example: preg=4
print("Prediction for preg=4:", pred_value)
