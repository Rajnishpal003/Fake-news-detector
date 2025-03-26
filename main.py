from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load trained model & vectorizer
with open("model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words) if words else None

def get_real_news_headlines():
    api_key = "fa010374bddd459b85a11ed3092264a8"  # Replace with your actual API key
    categories = ["general", "sports", "business", "technology", "entertainment"]
    headlines = []

    for category in categories:
        url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={api_key}"
        response = requests.get(url)
        data = response.json()
        fetched_headlines = [article["title"] for article in data.get("articles", [])]
        headlines.extend(fetched_headlines)

    print("\n📢 Fetched Real News Headlines:")  # Debugging
    for i, headline in enumerate(headlines[:5], 1):  # Print only first 5
        print(f"{i}. {headline}")
    
    if not headlines:
        print("⚠️ ERROR: No real news fetched! Check API key or NewsAPI response.")
    
    return headlines

def is_similar(text1, text2, threshold=0.8):  # Increased threshold for better accuracy
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > threshold

def has_significant_overlap(text1, text2, min_common_words=5):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    return len(words1 & words2) >= min_common_words

@app.route('/predict', methods=['POST'])
def predict_news():
    data = request.get_json()
    if not data or "texts" not in data or not isinstance(data["texts"], list):
        return jsonify({"error": "Invalid request. Please send a list of news texts."}), 400

    real_news = get_real_news_headlines()
    results = []

    for news_text in data["texts"]:
        processed_text = preprocess_text(news_text)

        for real in real_news:
            similarity_score = SequenceMatcher(None, news_text.lower(), real.lower()).ratio()
            print(f"🔍 Comparing: '{news_text}' ↔ '{real}' | Similarity: {similarity_score}")  # Debugging
            
            if similarity_score > 0.8 or has_significant_overlap(news_text, real):
                results.append({"text": news_text, "prediction": "Real (Matched with real news)"})
                break
        else:
            if processed_text is None:
                results.append({"text": news_text, "prediction": "Invalid (Please enter news)"})
                continue

            vectorized_text = vectorizer.transform([processed_text])
            prediction = classifier.predict(vectorized_text)
            prediction_proba = classifier.decision_function(vectorized_text)  # Get confidence scores
            print(f"🧠 ML Model Prediction for '{news_text}': {prediction[0]} (Confidence: {prediction_proba[0]})")  # Debugging
            
            prediction_label = "Real" if prediction[0] == 1 else "Fake"  # Flip labels if needed
            results.append({"text": news_text, "prediction": prediction_label})

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)