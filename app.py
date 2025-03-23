import streamlit as st
import pandas as pd
import joblib
import re
import string

# --- Load trained models and vectorizer ---
LR = joblib.load("LR_model.pkl")
DT = joblib.load("DT_model.pkl")
GB = joblib.load("GB_model.pkl")
RF = joblib.load("RF_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Text Cleaning Function ---
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Label decoder ---
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# --- Predict function ---
def manual_testing(news):
    new_data = pd.DataFrame({"text": [news]})
    new_data["text"] = new_data["text"].apply(wordopt)
    transformed = vectorizer.transform(new_data["text"])

    pred_LR = LR.predict(transformed)[0]
    pred_DT = DT.predict(transformed)[0]
    pred_GB = GB.predict(transformed)[0]
    pred_RF = RF.predict(transformed)[0]

    return {
        "Logistic Regression": output_label(pred_LR),
        "Decision Tree": output_label(pred_DT),
        "Gradient Boosting": output_label(pred_GB),
        "Random Forest": output_label(pred_RF),
    }

# --- Streamlit App UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste a news article below and click **Predict** to classify it using 4 ML models.")

news_input = st.text_area("✍️ Enter News Article:", height=300)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        results = manual_testing(news_input)
        st.subheader("🧠 Model Predictions:")
        for model, prediction in results.items():
            if prediction == "Fake News":
                st.error(f"{model}: {prediction}")
            else:
                st.success(f"{model}: {prediction}")
