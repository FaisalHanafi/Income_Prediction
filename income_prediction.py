# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv("C:\\Users\\Faisal\\Desktop\\LinkedIn\\Datasets\\adult_income.csv")

print (data)

col = data.columns

print(col)

# Replace missing values with the mean of the remaining values
data = data.fillna(data.mean())

# Remove duplicate rows
data = data.drop_duplicates()

# summarize the shape of the dataset
print(data.shape)
print(data.info)

# select columns with numerical data types
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
# select a subset of the dataframe with the chosen columns
subset = data[num_cols]

# create a histogram plot of each numeric variable
subset.hist()
pyplot.show()

data['income'] = data['income'].apply(lambda x: x.replace("<=50K", "0"))
data['income'] = data['income'].apply(lambda x: x.replace(">50K", "1"))
data['income'] = data['income'].astype(int)

# Split the data into features and labels
X = subset
y = data["income"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.2)

# Train a linear regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
print(model.score(X_test, y_test))

# Use the model to make predictions
predictions = model.predict(X_test)

print("The prediction is: ", predictions)

# Find the model's accuracy
accuracy_score(predictions, y_test.values)

cfm = confusion_matrix(predictions, y_test.values)
sns.heatmap(cfm, annot=True)

pyplot.xlabel('Predicted classes')
pyplot.ylabel('Actual classes')
pyplot.show()


