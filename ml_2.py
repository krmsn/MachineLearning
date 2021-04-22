import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3), "\n")

print(nyc.Date.values, "\n")

print(nyc.Date.values.reshape(-1, 1), "\n")

print(nyc.Temperature.values, "\n")

x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state = 11)

lr = LinearRegression()
lr.fit(X = x_train, y = y_train)

print(lr.coef_)
print(lr.intercept_)

predicted = lr.predict(x_test)

expected = y_test

# Will check for every FIFTH element in an array [::]
for p, e in zip(predicted[::5], expected[::5]):
    print("Predicted:", p)
    print("Expected:", e, "\n")

predict = (lambda x: lr.coef_ * x + lr.intercept_)

print(predict(2020))
print(predict(1890))
print(predict(2021), "\n")

axes = sns.scatterplot(data = nyc, x = "Date", y = "Temperature", hue = "Temperature", palette = "winter", legend = False)

axes.set_ylim(10,70)

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)

y = predict(x)

print(y)

line = plt.plot(x, y)

# Repeat process for yearly temperatures
print("(Repeat process for yearly temperatures)")
nyc_year = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")

print(nyc_year.head(3), "\n")

print(nyc_year.Date.values, "\n")

print(nyc_year.Date.values.reshape(-1, 1), "\n")

# This line cleans the "Date" column from the .csv file above - basically, it takes all the "12's out from the ends of the dates"
nyc_year.Date = nyc_year.Date.astype(str).str[:4].astype(int)

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(nyc_year.Date.values.reshape(-1, 1), nyc_year.Value.values, random_state = 11)

lr_1 = LinearRegression()
lr_1.fit(X = x_train_1, y = y_train_1)

print(lr_1.coef_)
print(lr_1.intercept_)

predicted_1 = lr_1.predict(x_test)

expected_1 = y_test_1

# Will check for every FIFTH element in an array [::]
for p, e in zip(predicted_1[::5], expected_1[::5]):
    print("Predicted:", p)
    print("Expected:", e, "\n")

predict = (lambda x: lr_1.coef_ * x + lr_1.intercept_)

print(predict(2020))
print(predict(1890))
print(predict(2021), "\n")

axes = sns.scatterplot(data = nyc_year, x = "Date", y = "Value", hue = "Value", palette = "winter", legend = False)

axes.set_ylim(10,70)

x_1 = np.array([min(nyc_year.Date.values), max(nyc_year.Date.values)])
print(x)

y_1 = predict(x_1)

print(y_1)

line_1 = plt.plot(x_1, y_1)
plt.show()