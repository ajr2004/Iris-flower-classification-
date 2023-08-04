import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Load the dataset
df = pd.read_csv('Iris.csv')

# Explore the dataset
print("Iris Dataset:")
print(df.head())
print("\nDescriptive statistics of the dataset:")
print(df.describe())

# Count the occurrences of each species
print("\nSpecies value counts:")
print(df["Species"].value_counts())

# Visualization - Scatter plot with hue (Species)
plt.figure(figsize=(10, 7))
sns.scatterplot(x="PetalLengthCm", y="SepalWidthCm", hue="Species", data=df, s=100)
plt.title("Petal Length vs Sepal Width (with Species)")
plt.legend(title="Species")
plt.show()

# Pairplot to visualize relationships between features with hue (Species)
sns.pairplot(df, hue="Species")
plt.show()

# Prepare the data for training
data = df.values
x = data[:, 0:5]
y = data[:, 5]

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Model training - Logistic Regression
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(x_train, y_train)

# Prediction on test data
prediction = model_LR.predict(x_test)

# Evaluation - Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction) * 100
print("\nAccuracy of the model: {:.2f}%".format(accuracy))

# Classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, prediction)
print("\nClassification Report:\n", report)

# New data points for prediction
x_new = np.array([[5, 3, 2, 1, 0.2], [145, 1.9, 2.2, 3.8, 1.1], [3.5, 3.2, 2.5, 4.6, 1.9]])
prediction_new = model_LR.predict(x_new)
print("\nPrediction of species for new data points:")
print(prediction_new)
