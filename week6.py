# Loading data with CSV package - basic reading
import csv

with open("/Users/devadarsanna/Downloads/example2.csv") as file:
    reader = csv.reader(file)
    for col in reader:
        print(col)

# CSV package - pulling data by columns
import csv

with open("/Users/devadarsanna/Downloads/example2.csv") as file:
    reader = csv.reader(file)
    for col in reader:
        print(col[0], col[1], col[2], col[3])

# NumPy - loading with genfromtxt (numeric values)
from numpy import genfromtxt

data = genfromtxt("/Users/devadarsanna/Downloads/example2.csv", delimiter=",")
print(data)

# NumPy - loading with genfromtxt (string values)
from numpy import genfromtxt

data = genfromtxt("/Users/devadarsanna/Downloads/example2.csv", delimiter=",", dtype=str)
print(data)

# NumPy - loading larger dataset and checking shape
from numpy import genfromtxt

data = genfromtxt("/Users/devadarsanna/Downloads/pima_indians_diabetes.csv", delimiter=",")
print(data)
print(data.shape)

# Pandas - basic CSV reading
import pandas as pd

data = pd.read_csv("/Users/devadarsanna/Downloads/example2.csv")
print(data)

# Pandas - loading larger dataset
import pandas as pd

data = pd.read_csv("/Users/devadarsanna/Downloads/pima_indians_diabetes.csv")
print(data.head())

# Pandas - loading with custom column headers
import pandas as pd

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("/Users/devadarsanna/Downloads/pima_indians_diabetes.csv", names=names)
print(data.head())

# View first 5 rows
print(data.head())

# View last 5 rows
print(data.tail())

# Converting NumPy array to Pandas DataFrame
from numpy import genfromtxt
import pandas as pd

data_array = genfromtxt("/Users/devadarsanna/Downloads/pima_indians_diabetes.csv", delimiter=",")

# Convert to DataFrame
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.DataFrame(data_array, columns=names)

# Calculate mean
print(data.mean())

# Calculate median
print(data.median())

# Calculate mode
print(data.mode())

# Overall statistics summary
print(data.describe())

# Loading dataset with null values
import pandas as pd

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv("/Users/devadarsanna/Downloads/pima_indians_diabetes_2.csv", names=names)
print(data.shape)

# Checking for null values
print(data.isnull().sum())
print(data[data.isnull().any(axis=1)])

# With skipna=False, variables with null values aren't calculated
print(data.mean(skipna=False))

# Initial shape before dropping null values
print("Initial shape:", data.shape)

# Drop rows with any null values
data_clean = data.dropna()
print("New shape:", data_clean.shape)

# Calculate mean on cleaned data
print(data_clean.mean())

# Handling null values - imputation with median strategy
from sklearn.impute import SimpleImputer
import numpy as np

# Create imputer with median strategy
imputer = SimpleImputer(strategy='median')

# Fit and transform the data
data_imputed = imputer.fit_transform(data)

# Convert back to DataFrame
data_filled = pd.DataFrame(data_imputed, columns=names)

# Example with mean strategy
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
data_filled = pd.DataFrame(data_imputed, columns=names)
print(data_filled.mean())

# Viewing imputed data results
print("Shape after imputation:", data_filled.shape)
print(data_filled.head(10))
print(data_filled.mean())