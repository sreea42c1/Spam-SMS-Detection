# Spam-SMS-Detection
Spam SMS detection is the process of identifying and filtering out unwanted or fraudulent text messages. These messages often contain advertisements, scams, phishing attempts, or malicious links. Effective spam detection helps protect users from fraud and improves their overall messaging experience.


import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset (SMS Spam Collection Dataset)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms-spam-collection.csv"
data = pd.read_csv(url, encoding='latin-1')
data.columns = ['label', 'message']

# Convert Labels to Binary (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Data Preprocessing Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    return text

# Apply Cleaning
data['message'] = data['message'].apply(clean_text)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Build Pipeline (Vectorization + Naïve Bayes)
spam_detector = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train Model
spam_detector.fit(X_train, y_train)

# Predictions
y_pred = spam_detector.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to Predict New Messages
def predict_spam(message):
    return "Spam" if spam_detector.predict([clean_text(message)])[0] == 1 else "Ham"

# Test Example
print(predict_spam("Congratulations! You won a free lottery. Claim now!"))
