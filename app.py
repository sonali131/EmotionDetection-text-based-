import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="centered"
)

# -------------------------
# Load PKL Files
# -------------------------
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("emotion_model.pkl", "rb"))
label_map = pickle.load(open("label_map.pkl", "rb"))

# -------------------------
# Styling (CSS)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #f9fafc;
}
.title-container {
    background: linear-gradient(135deg, #1b0033, #3a004f, #0f172a);
    padding: 30px 20px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0px 12px 30px rgba(0, 0, 0, 0.4);
}

.title-text {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #f9fafb;
    letter-spacing: 1.2px;
    text-shadow: 3px 3px 10px rgba(0,0,0,0.7);
}

.subtitle-text {
    text-align: center;
    font-size: 18px;
    color: #e5e7eb;
    margin-top: 8px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #eef2ff;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #1f2937;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header Section
# -------------------------
st.markdown("<div class='title-text'>😊 Emotion Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>AI-powered text emotion analysis using Machine Learning</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/742/742751.png",
        width=120
    )
    st.title("About Project")
    st.write("""
    🔹 **Model:** Linear SVM  
    🔹 **Vectorizer:** TF-IDF  
    🔹 **Dataset:** Emotion Text Dataset  
    🔹 **Accuracy:** ~80%  

    Built using **Python & Streamlit**
    """)

# -------------------------
# Preprocessing
# -------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = ''.join([i for i in txt if i.isascii()])
    words = txt.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------
# Emotion Emoji Mapping
# -------------------------
emotion_emoji = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# -------------------------
# Input Section
# -------------------------
st.subheader("✍️ Enter Your Text")
user_text = st.text_area(
    "",
    placeholder="Example: I am feeling very happy today!",
    height=120
)

# -------------------------
# Prediction
# -------------------------
if st.button("🎯 Predict Emotion"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = clean_text(user_text)
        vector = tfidf.transform([cleaned])
        pred = model.predict(vector)[0]
        emotion = label_map[pred]

        emoji = emotion_emoji.get(emotion.lower(), "🙂")

        st.markdown(
            f"<div class='result-box'>Predicted Emotion: {emoji} <br> {emotion.upper()}</div>",
            unsafe_allow_html=True
        )

        st.balloons()

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("<div class='footer'>🚀 Emotion Detection App | Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
