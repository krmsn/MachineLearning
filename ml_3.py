from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import seaborn as sns

cali = fetch_california_housing()   # bunch object
# print(cali.DESCR)
print(cali.data.shape, "\n")
# The .target.shape method calls by one feature
print(cali.target.shape, "\n")
print(cali.feature_names, "\n")

pd.set_option("precision", 4)
pd.set_option("max_columns", 9)
pd.set_option("display.width", None)

cali_df = pd.DataFrame(cali.data, columns = cali.feature_names)

cali_df["MedHouseValue"] = pd.Series(cali.target)
print(cali_df.head(), "\n")

sample_df = cali_df.sample(frac = 0.1)

sns.set(font_scale = 2)
sns.set_style("whitegrid")

for feature in cali.feature_names:
    plt.figure(figsize = (8, 4.5)) # 8" by 4.5" figure
    sns.scatterplot(
        data = sample_df,
        x = feature,
        y = "MedHouseValue",
        hue = "MedHouseValue",
        palette = "cool",
        legend = False
    )

# plt.show()

train_data, test_data, train_target, test_target = train_test_split(cali.data, cali.target, random_state = 11)

lr = LinearRegression()
# Tells the model the training target for each row of training data.
lr.fit(X = train_data, y = train_target)

predicted = lr.predict(test_data)
expected = test_target
print("Predicted:", predicted[::5], "\n")
print("Expected:", expected[::5], "\n")

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

figure = plt2.figure(figsize = (9, 9))

axes = sns.scatterplot(
    data = df,
    x = "Expected",
    y = "Predicted",
    hue = "Predicted",
    palette = "cool",
    legend = False
)

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
print(start)
print(end)

axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt2.plot([start, end], [start, end], "k--")
# plt2.show()

sns.set(font_scale = 1.1)
sns.set_style("whitegrid")
grid = sns.pairplot(data = cali_df, vars = cali_df.columns[0:4])

plt.show()
