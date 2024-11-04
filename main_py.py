# Import all necessary libraries
import pandas as pd
import numpy as np

# Load the CSV files
file_path_1 = 'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv'
file_path_2 = 'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'

# Load the datasets into pandas DataFrames
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)

# Create a function that transforms the review ratings into Positive, Neutral, or Negative categories
def categorize_rating(rating):
    if rating == 5:
        return 'Positive'
    elif rating == 4:
        return 'Neutral'
    elif rating in [1, 2, 3]:
        return 'Negative'
    else:
        return None
    
# Apply the function to reviews.rating column and create a new column called 'sentiment' in both datasets
df1['sentiment'] = df1['reviews.rating'].apply(categorize_rating)
df2['sentiment'] = df2['reviews.rating'].apply(categorize_rating)

# Check for missing values in both DataFrames
missing_values_df1 = pd.isnull(df1)
missing_values_df2 = pd.isnull(df2)

# Count missing values in each column
missing_counts_df1 = missing_values_df1.sum()
missing_counts_df2 = missing_values_df2.sum()

# Calculate the total number of missing values
total_missing_values_df1 = missing_counts_df1.sum()
total_missing_values_df2 = missing_counts_df2.sum()

# Display the results
print("\nTotal Missing Values in the DataFrame 1:", total_missing_values_df1)
print("\nTotal Missing Values in the DataFrame 2:", total_missing_values_df2)

# Since there are columns with a great number of missing values that we don't need for the NLP exercise, now let's filter the essential columns 
df1_new = df1[['categories', 'reviews.rating', 'reviews.text', 'sentiment']]
df2_new = df2[['categories', 'reviews.rating', 'reviews.text', 'sentiment']]


# Data cleaning

import re
import string

# Define a function to preprocess the text data
def preprocess_text(text):
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove special characters and punctuation using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

# Apply the text preprocessing function to both datasets
df1_new['cleaned_text'] = df1_new['reviews.text'].apply(preprocess_text)
df2_new['cleaned_text'] = df2_new['reviews.text'].apply(preprocess_text)


# Tokenization and Lemmatization

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() # Initialize the lemmatizer

# Define a function to tokenize and lemmatize the text
def tokenize_and_lemmatize(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a single string
    return ' '.join(lemmatized_tokens)

# Apply the function to the 'cleaned_text' column in both datasets
df1_new['lemmatized_text'] = df1_new['cleaned_text'].apply(tokenize_and_lemmatize)
df2_new['lemmatized_text'] = df2_new['cleaned_text'].apply(tokenize_and_lemmatize)

# In the end we are just using the df2_new dataset to train our models

# Separate the data into features and labels
X = df2_new['lemmatized_text'] 
y = df2_new['sentiment'] 

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Apply the TF-IDF Vectorizer to the 'lemmatized_text' column
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Model Building

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

# Initialize the Multinomial Naive Bayes model
nb_model = MultinomialNB()

# Define the hyperparameters grid for alpha (smoothing parameter)
param_grid = {'alpha': [0.1, 0.2]}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search_nb = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data
grid_search_nb.fit(X_train_tfidf, y_train)

# Output the best parameters found
print(f"Best Parameters: {grid_search_nb.best_params_}")

# Train the model using the best hyperparameters
best_nb_model = grid_search_nb.best_estimator_

# Evaluate the model on the test set
y_pred_nb = best_nb_model.predict(X_test_tfidf)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"Multinomial Naive Bayes Accuracy with Best Parameters: {accuracy_nb:.4f}")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the confusion matrix
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb, labels=['Positive', 'Neutral', 'Negative'])

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
lr_model = LogisticRegression()

# Define the hyperparameters grid for tuning (regularization parameter C)
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}  # C: controls the strength of regularization # Penalty specifies the type L1 and L2 regularization

# Initialize GridSearchCV with 5-fold cross-validation
grid_search_lr = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data
grid_search_lr.fit(X_train_tfidf, y_train)

# Output the best parameters found
print(f"Best Parameters: {grid_search_lr.best_params_}")

# Train the model using the best hyperparameters
best_lr_model = grid_search_lr.best_estimator_

# Evaluate the model on the test set
y_pred_lr = best_lr_model.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"Logistic Regression Accuracy with Best Parameters: {accuracy_lr:.4f}")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Calculate and plot the confusion matrix
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr, labels=['Positive', 'Neutral', 'Negative'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lr, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


from sklearn.svm import SVC

# Initialize the Support Vector Classifier
svm_model = SVC()

# Define the hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10],             # Regularization parameter
    'kernel': ['linear', 'rbf'],    # Kernel type: linear or radial basis function (rbf)
    'gamma': ['scale', 'auto']      # Kernel coefficient (only used for 'rbf' kernel)
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data
grid_search_svm.fit(X_train_tfidf, y_train)

# Output the best parameters found
print(f"Best Parameters: {grid_search_svm.best_params_}")

# Train the model using the best hyperparameters
best_svm_model = grid_search_svm.best_estimator_

# Evaluate the model on the test set
y_pred_svm = best_svm_model.predict(X_test_tfidf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Support Vector Machine Accuracy with Best Parameters: {accuracy_svm:.4f}")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Calculate and plot the confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm, labels=['Positive', 'Neutral', 'Negative'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title("Confusion Matrix - Support Vector Machine")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [10, 20, None],     # Depth of each tree
    'min_samples_split': [2, 5, 10]  # Minimum samples required to split a node
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Train the model on the training data
grid_search_rf.fit(X_train_tfidf, y_train)

# Output the best parameters found
print(f"Best Parameters: {grid_search_rf.best_params_}")

# Train the model using the best hyperparameters
best_rf_model = grid_search_rf.best_estimator_

# Evaluate the model on the test set
y_pred_rf = best_rf_model.predict(X_test_tfidf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy with Best Parameters: {accuracy_rf:.4f}")

# Display a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Calculate and plot the confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf, labels=['Positive', 'Neutral', 'Negative'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
