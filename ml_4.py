from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Import .csv files and convert them to DataFrames
animal_classes = pd.read_csv("animal_classes.csv")
animals_train = pd.read_csv("animals_train.csv")
animals_test = pd.read_csv("animals_test.csv")
# Print DataFrames
print(animal_classes)
print(animals_train)
print(animals_test)

# # # For each animal feature set, we must predict the animal class number
# First, let's slice our DataFrames into the training and testing data we need
train_target = animals_train.class_number.values.reshape(-1)
test_target = animals_test.animal_name.values.reshape(-1)

train_data = animals_train.drop("class_number", axis = 1)
train_data = train_data.values

test_data = animals_test.drop("animal_name", axis = 1)
test_data = test_data.values

# Print data, targets and their shapes - confirm they are a correct fit for KNearestNeighbors()
print(train_data)
print(test_data, "\n")
print(train_data.shape)
print(test_data.shape, "\n")
print(train_target, "\n")
print(train_target.shape, "\n")

# We use KNearestNeighbors() to fit our training dataset for classification
# with ".predict()," a machine learning algorithm generates targets for our testing data
knn = KNeighborsClassifier()
knn.fit(train_data, train_target)
animal_class_number = knn.predict(test_data)

print(animal_class_number, "\n")

# Translate the machine's "Class_Number" predictions into their "Class_Type" names
animal_class_type = [animal_classes.at[a - 1, "Class_Type"] for a in animal_class_number]
print(animal_class_type, "\n")

# Generate the animal kingdom
kingdom = {"animal_name": list(animals_test.animal_name.values), "prediction": animal_class_type}
print(kingdom, "\n")

# Convert the kingdom into a dataframe for our final .csv
predictions = pd.DataFrame(kingdom)
print(predictions, "\n")

# Send "predictions" DataFrame to .csv file - "km_predictions.csv"
predictions.to_csv(r"km_predictions.csv", index = False)

# For future reference:
# x_train = training data
# y_train = training target
# x_test = testing data
# y_test = testing target
