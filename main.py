from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Load necessary NLP datasets
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)
CORS(app)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters & numbers
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    # Check if the processed input is empty or too short (not meaningful news)
    if len(words) < 3:  # Less than 3 words means it's likely not real news
        return None  # Return None so we can detect invalid input in Flask

    return ' '.join(words)



@app.route('/predict', methods=['POST'])
def predict_news():
    data = request.get_json()

    # Check if the request contains a list of news texts
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Invalid request. Please send a list of news texts."}), 400

    results = []  # Store results for each news text

    for news_text in data["texts"]:
        processed_text = preprocess_text(news_text)

        # If preprocessing returns None, it's not valid news
        if processed_text is None:
            results.append({"text": news_text, "prediction": "Invalid (Please enter news)"})
            continue  # Skip this news entry and move to the next one

        vectorized_text = vectorizer.transform([processed_text])
        prediction = classifier.predict(vectorized_text)

        prediction_label = "Fake" if prediction[0] == 1 else "Real"
        results.append({"text": news_text, "prediction": prediction_label})

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(debug=True)
