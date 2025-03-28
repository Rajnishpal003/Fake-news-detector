import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Replace PassiveAggressiveClassifier with Logistic Regression
classifier = LogisticRegression(max_iter=1000)  # Increased iterations for accuracy
classifier.fit(X_train, y_train)



# Download stopwords if not downloaded
nltk.download('stopwords')

# Load the dataset (Ensure you have 'news.csv' in your project folder)
df = pd.read_csv("news.csv")  # Make sure this file exists!

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df["text"] = df["text"].apply(preprocess_text)

# Split data
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
classifier = PassiveAggressiveClassifier()
classifier.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")
