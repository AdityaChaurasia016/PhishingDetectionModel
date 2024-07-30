import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re

# Load dataset
data = pd.read_csv('emails.csv')
X = data['text']
y = data['label']

# Feature extraction
def extract_features(email_text):
    features = {}
    features['num_links'] = len(re.findall(r'http[s]?://', email_text))
    features['html_content'] = int(bool(re.search(r'<[^>]+>', email_text)))
    features['num_suspicious_words'] = len(re.findall(r'\b(password|bank|login|click|account|verify)\b', email_text.lower()))
    return features

# Vectorize email text using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Add new features to the dataset
additional_features = pd.DataFrame([extract_features(text) for text in X])
X_combined = pd.concat([pd.DataFrame(X_tfidf.toarray()), additional_features], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def detect_phishing(email_text):
    email_tfidf = vectorizer.transform([email_text])
    email_features = pd.DataFrame([extract_features(email_text)])
    email_combined = pd.concat([pd.DataFrame(email_tfidf.toarray()), email_features], axis=1)
    prediction = model.predict(email_combined)
    return 'Phishing' if prediction[0] == 1 else 'Legitimate'
