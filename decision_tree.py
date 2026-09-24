# Decision Tree Classification using Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Load Iris dataset
iris = load_iris()

# 2. Separate input features and target classes
X = iris.data
y = iris.target

# 3. Display dataset information
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Feature names:", iris.feature_names)
print("Class names:", iris.target_names)

# 4. Split dataset - 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create Decision Tree classifier using Gini criterion
model = DecisionTreeClassifier(criterion="gini", random_state=42)

# 6. Train the model
model.fit(X_train, y_train)

#7. Predict test data
y_pred = model.predict(X_test)

# 8. Display actual and predicted classes
print("\nActual Classes:")
print(y_test)

print("\nPredicted Classes:")
print(y_pred)

# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy (80:20):", accuracy)

# 10. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 12 & 13. Visualize the Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.title("Decision Tree - Iris Dataset")
plt.show()


# Try 70% training and 30% testing
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model2 = DecisionTreeClassifier(criterion="gini", random_state=42)
model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_test2)

accuracy2 = accuracy_score(y_test2, y_pred2)

print("\nAccuracy (70:30):", accuracy2)
print("\nConfusion Matrix (70:30):")
print(confusion_matrix(y_test2, y_pred2))

print("\nClassification Report (70:30):")
print(classification_report(
    y_test2, y_pred2, target_names=iris.target_names
))