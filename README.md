# Fake News Detector

## 📌 Overview
This is a **Fake News Detector** that uses **Machine Learning (ML) and Real-Time Fact-Checking** to classify news as **Real** or **Fake**.

- ✅ **ML Model:** Classifies news based on text analysis
- ✅ **Real-Time Fact-Checking:** Fetches headlines from trusted news sources via **NewsAPI**
- ✅ **Flask API:** Backend for processing news inputs
- ✅ **Frontend Support:** Can be integrated with a web UI

---

## 🚀 Features
- **Detect Fake News** using an ML model
- **Fact-check with Real-Time News** (BBC, CNN, Reuters, etc.)
- **Confidence Score** for predictions
- **Logs & Debugging Support**

---

## 🛠️ Tech Stack
- **Python** (Flask, Scikit-Learn, NLTK, Requests)
- **Machine Learning** (TF-IDF + Classifier Model)
- **NewsAPI** (For real-time fact-checking)
- **Postman** (For API Testing)

---

## 📌 Installation
### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Key
- Sign up at [NewsAPI](https://newsapi.org/)
- Replace `YOUR_NEWSAPI_KEY` in `main.py` with your API key

### 4️⃣ Run Flask Server
```bash
python main.py
```
Server will run on `http://127.0.0.1:5000`

---

## 🔥 How to Use
### **Using API (Postman)**
- **POST Request to `/predict`**
- **Body (JSON Format):**
```json
{
  "texts": [
    "Breaking: Scientists discover a new planet with signs of life."
  ]
}
```
- **Example Response:**
```json
{
  "results": [
    {"text": "Breaking: Scientists discover a new planet with signs of life.", "prediction": "Real (Matched with real news)"}
  ]
}
```

---

## 📌 To-Do / Future Improvements
- ✅ **Improve ML Model Accuracy** (More training data)
- ✅ **Enhance UI with React or Vue**
- ✅ **Deploy to Heroku or AWS**
- ✅ **Allow User Feedback for Model Retraining**

---

## 📝 License
This project is **open-source** and available under the MIT License.

📢 **Contributions Welcome!** Feel free to fork & improve! 🚀

