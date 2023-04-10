import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('credit_card_transactions.csv')

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df.drop('fraud', axis=1), df['fraud'], test_size=0.2, random_state=42)

# Train the random forest classifier on the training data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Use the trained classifier to make predictions on the testing data
y_pred = clf.predict(X_test)

# Print the accuracy of the classifier on the testing data
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the confusion matrix of the classifier on the testing data
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
