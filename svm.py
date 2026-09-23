# Text Classification using SVM
# Dataset: 20 Newsgroups Dataset

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Select the news categories to be used for classification
categories = ['comp.graphics', 'rec.sport.baseball', 'sci.space']

# Load the training portion of the 20 Newsgroups dataset
train_data = fetch_20newsgroups(subset='train', categories=categories,
                                 remove=('headers', 'footers', 'quotes'))

# Load the testing portion of the 20 Newsgroups dataset
test_data = fetch_20newsgroups(subset='test', categories=categories,
                                remove=('headers', 'footers', 'quotes'))

# Create a TfidfVectorizer to convert text documents into numerical TF-IDF feature vectors
vectorizer = TfidfVectorizer()

# Fit the TF-IDF vectorizer on the training documents and transform them
X_train = vectorizer.fit_transform(train_data.data)

# Transform the test documents using the same TF-IDF vectorizer
X_test = vectorizer.transform(test_data.data)

# Separate the target class labels for training and testing datasets
y_train = train_data.target
y_test = test_data.target

# Create a Linear Support Vector Machine classifier
svm_classifier = LinearSVC()

# Train the SVM model using the TF-IDF training features and their labels
svm_classifier.fit(X_train, y_train)

# Use the trained SVM model to predict the classes of the test documents
y_pred = svm_classifier.predict(X_test)

# Calculate the classification accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report showing precision, recall, F1-score, and support
report = classification_report(y_test, y_pred, target_names=test_data.target_names)

# Display the number of training and testing documents, accuracy, and classification report
print(f"Number of training documents: {len(train_data.data)}")
print(f"Number of testing documents: {len(test_data.data)}")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", report)
