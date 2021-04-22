from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

# # # This is an example of "classification" machine learning - using classes for numbers "0" to "9"

digits = load_digits()

# print(digits.DESCR)

# print(digits.data[:2], "\n")
# print(digits.data.shape, "\n")
# print(digits.target[:2], "\n")
# print(digits.target.shape, "\n")

print(digits.images[:2], "\n")

fig, axes = plt.subplots(nrows = 4, ncols = 6, figsize = (6, 4))

# Python "zip" function bundles
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap = plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)


plt.tight_layout()

# plt.show()

# x_train houses all our sample data
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 11)

print(digits.data[:10], "\n")
print(digits.target[:10], "\n")

print(x_train[:10])
print(y_train[:10], "\n")

print(x_test[:10])
print(y_test[:10], "\n")

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

knn = KNeighborsClassifier()

# Load the training data into the model using the "fit" method
knn.fit(X = x_train, y = y_train)

predicted = knn.predict(X = x_test)
expected = y_test

print(predicted[:20])
print(expected[:20], "\n")

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong, "\n")

print(format(knn.score(x_test, y_test), ".2%"), "\n")

cf = confusion_matrix(y_true = expected, y_pred = predicted)
print(cf, "\n")

cf_df = pd.DataFrame(cf, index = range(10), columns = range(10))

fig = plt2.figure(figsize = (7, 6))
axes = sns.heatmap(cf_df, annot = True, cmap = plt2.cm.nipy_spectral_r)
plt2.show()
