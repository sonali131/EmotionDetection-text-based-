
# import pandas as pd
# import string
# import nltk
# import pickle

# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # -----------------------------
# # Load Dataset
# # -----------------------------
# df = pd.read_csv(
#     "train.txt",
#     sep=";",
#     header=None,
#     names=["text", "emotion"]
# )

# # -----------------------------
# # FIXED Emotion Order (VERY IMPORTANT)
# # -----------------------------
# emotion_list = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise','neutral']

# emotion_numbers = {emo: i for i, emo in enumerate(emotion_list)}
# num_to_emotion = {i: emo for emo, i in emotion_numbers.items()}

# df['emotion'] = df['emotion'].map(emotion_numbers)

# # -----------------------------
# # Text Preprocessing
# # -----------------------------
# df['text'] = df['text'].str.lower()

# def remove_punctuation(txt):
#     return txt.translate(str.maketrans('', '', string.punctuation))

# def remove_numbers(txt):
#     return ''.join([c for c in txt if not c.isdigit()])

# def remove_non_ascii(txt):
#     return ''.join([c for c in txt if c.isascii()])

# df['text'] = df['text'].apply(remove_punctuation)
# df['text'] = df['text'].apply(remove_numbers)
# df['text'] = df['text'].apply(remove_non_ascii)

# # -----------------------------
# # Stopwords (Emotion-Safe)
# # -----------------------------
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# custom_stopwords = stop_words - {
#     "not", "no", "never", "very",
#     "unexpected", "suddenly",
#     "shock", "shocked", "surprised"
# }

# def remove_stopwords(txt):
#     return " ".join([w for w in txt.split() if w not in custom_stopwords])

# df['text'] = df['text'].apply(remove_stopwords)

# # -----------------------------
# # Train-Test Split
# # -----------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text'],
#     df['emotion'],
#     test_size=0.20,
#     random_state=42,
#     stratify=df['emotion']
# )

# # -----------------------------
# # TF-IDF Vectorizer (Improved)
# # -----------------------------
# tfidf_vectorizer = TfidfVectorizer(
#     ngram_range=(1, 2),
#     min_df=2,
#     max_df=0.95
# )

# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# X_test_tfidf = tfidf_vectorizer.transform(X_test)

# # -----------------------------
# # Logistic Regression (Balanced)
# # -----------------------------
# emotion_model = LogisticRegression(
#     max_iter=1000,
#     class_weight="balanced"
# )

# emotion_model.fit(X_train_tfidf, y_train)

# # -----------------------------
# # Save PKL Files
# # -----------------------------
# pickle.dump(emotion_model, open("emotion_model.pkl", "wb"))
# pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
# pickle.dump(num_to_emotion, open("label_map.pkl", "wb"))

# print("✅ Model trained & PKL files saved successfully!")

import pandas as pd
import string
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(
    "train.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

# -----------------------------
# FIXED Emotion Order (VERY IMPORTANT)
# -----------------------------
emotion_list = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise','neutral']

emotion_numbers = {emo: i for i, emo in enumerate(emotion_list)}
num_to_emotion = {i: emo for emo, i in emotion_numbers.items()}

df['emotion'] = df['emotion'].map(emotion_numbers)

# -----------------------------
# Text Preprocessing
# -----------------------------
df['text'] = df['text'].str.lower()

def remove_punctuation(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(txt):
    return ''.join([c for c in txt if not c.isdigit()])

def remove_non_ascii(txt):
    return ''.join([c for c in txt if c.isascii()])

df['text'] = df['text'].apply(remove_punctuation)
df['text'] = df['text'].apply(remove_numbers)
df['text'] = df['text'].apply(remove_non_ascii)

# -----------------------------
# Stopwords (Emotion-Safe)
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

custom_stopwords = stop_words - {
    "not", "no", "never", "very",
    "unexpected", "suddenly",
    "shock", "shocked", "surprised"
}

def remove_stopwords(txt):
    return " ".join([w for w in txt.split() if w not in custom_stopwords])

df['text'] = df['text'].apply(remove_stopwords)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['emotion'],
    test_size=0.20,
    random_state=40,
    stratify=df['emotion']
)

# -----------------------------
# TF-IDF Vectorizer (Improved)
# -----------------------------
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# -----------------------------
# Logistic Regression (Balanced)
# -----------------------------
emotion_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

emotion_model.fit(X_train_tfidf, y_train)

# -----------------------------
# Save PKL Files
# -----------------------------
pickle.dump(emotion_model, open("emotion_model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(num_to_emotion, open("label_map.pkl", "wb"))

print("✅ Model trained & PKL files saved successfully!")
