# EmotionDetection-text-based
DataSet+NLTK+Streamlit
# 😊 Emotion Detection Web App

[🔗 Live Demo](https://wuqidvhoavjnqgg6yatfou.streamlit.app/)  
A **deployed Streamlit app** that detects the emotion behind user-entered text using Machine Learning.

---

## 📌 Project Description

This is an **AI-powered text emotion detection system** that analyzes text input and predicts the expressed emotion using a trained machine learning model.

The app supports the following emotions:

- 😄 **Joy**
- 😢 **Sadness**
- 😡 **Anger**
- 😨 **Fear**
- ❤️ **Love**
- 😲 **Surprise**
- 😐 **Neutral**

The model uses **Natural Language Processing (NLP)** techniques and a machine learning classifier to analyze text and return a predicted emotion.

---

## 📺 Live App

Check out the deployed app here:

👉 https://wuqidvhoavjnqgg6yatfou.streamlit.app/

Feel free to enter text and see real-time emotion predictions!

---

## 🧠 How It Works

1. User enters text in the input box.
2. Text is cleaned using NLP preprocessing (lowercasing, punctuation removal, etc.).
3. **TF-IDF** (Term Frequency–Inverse Document Frequency) converts text to numerical vectors.
4. A trained ML model predicts the most likely emotion.
5. The app displays the predicted emotion with an emoji 🔥

---

## 📁 Project Structure

EmotionDetection-text-based-
│
├── app.py # Streamlit web UI
├── train_and_save_model.py # Script that trains and saves model + vectorizer
├── train.txt # Emotion labeled dataset
├── emotion_model.pkl # Saved trained ML model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── label_map.pkl # Mapping from numeric labels to labels
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 How to Run Locally

If you want to run the app on your local machine:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sonali131/EmotionDetection-text-based-.git
cd EmotionDetection-text-based-
2️⃣ Install Dependencies
Make sure you have Python installed, then:

bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
Your browser will open the app at http://localhost:8501.

🛠️ Dependencies
This project uses:

Python

Streamlit

NLTK

Scikit-learn

Pandas

Example requirements.txt:

nginx
Copy code
streamlit
nltk
scikit-learn
pandas
🧪 Sample Inputs
Input Text	Predicted Emotion
“I’m so happy today!”	😄 Joy
“I feel so sad and alone.”	😢 Sadness
“I am scared of exams.”	😨 Fear


📊 Model Details
Vectorizer: TF-IDF

Algorithm: Logistic Regression / Linear SVM

Emotion classes: anger, fear, joy, love, sadness, surprise, neutral

Dataset: Emotion Text Dataset

Approx Accuracy: ~80%
## 🖼️ UI Screenshot

![Emotion Detection App UI](<img width="944" height="436" alt="image" src="https://github.com/user-attachments/assets/3253bbba-2c22-4b31-92cf-e9394d895948" />
)


💡 Notes
✔ This app provides a baseline emotion detection model.
✔ For better accuracy, you can retrain the model with more data, advanced preprocessing, or a deep learning model (e.g., BERT).

This project is designed so that you can improve, extend, or retrain the model based on your needs.

🧠 Future Improvements
Use BERT / Transformer models for context-aware emotion detection

Add confidence scores

Add speech-to-text emotion prediction

Support multiple languages

🤝 Contribution
Feel free to open issues, submit pull requests, or request new features!
Happy coding 😊

👩‍💻Author
Sonali Mishra
GitHub: https://github.com/sonali131
