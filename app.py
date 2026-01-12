import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="🎬 AI Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ===================== THEME TOGGLE =====================
dark_mode = st.toggle("🌗 Dark / Light Mode", value=True)

bg = (
    "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
    if dark_mode
    else "linear-gradient(135deg, #f5f7fa, #c3cfe2)"
)

text_color = "white" if dark_mode else "#111827"

# ===================== CUSTOM CSS =====================
st.markdown(f"""
<style>
body {{
    background: {bg};
    color: {text_color};
}}
.card {{
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}}
.title {{
    font-size: 40px;
    font-weight: 800;
    text-align: center;
}}
.subtitle {{
    text-align: center;
    opacity: 0.85;
    margin-bottom: 30px;
}}
footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

# ===================== SAFETY CHECK =====================
if not os.path.exists("simple_rnn_imdb.h5"):
    st.error("❌ Model file not found: simple_rnn_imdb.h5")
    st.stop()

# ===================== LOAD MODEL & WORD INDEX =====================
@st.cache_resource
def load_model_and_vocab():
    model = tf.keras.models.load_model("simple_rnn_imdb.h5")
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_model_and_vocab()

MAX_LEN = 500

# ===================== PREPROCESS FUNCTION =====================
def preprocess_text(text):
    words = text.lower().strip().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    padded = sequence.pad_sequences([encoded], maxlen=MAX_LEN)
    return padded, len(encoded)

# ===================== LIGHTWEIGHT EXPLANATION =====================
NEGATIVE_WORDS = {"worst", "bad", "terrible", "awful", "boring", "hate"}
POSITIVE_WORDS = {"amazing", "great", "fantastic", "excellent", "love"}

def explain_sentence(text):
    words = set(text.lower().split())
    explanations = []

    if words & NEGATIVE_WORDS:
        explanations.append("⚠️ Negative emotional words detected")
    if words & POSITIVE_WORDS:
        explanations.append("✅ Positive emotional words detected")

    return explanations if explanations else ["ℹ️ No strong emotional keywords detected"]

# ===================== SESSION HISTORY =====================
if "history" not in st.session_state:
    st.session_state.history = []

# ===================== HEADER =====================
st.markdown("<div class='title'>🎬 AI Movie Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Simple RNN + Embedding | End-to-End NLP System</div>",
    unsafe_allow_html=True
)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # ===================== EXAMPLE BUTTONS =====================
    colA, colB, colC = st.columns(3)

    if colA.button("😊 Positive Example"):
        st.session_state.example = "This movie was absolutely amazing and fantastic"

    if colB.button("😞 Negative Example"):
        st.session_state.example = "Worst movie I have ever watched boring and awful"

    if colC.button("😐 Ambiguous Example"):
        st.session_state.example = "The movie was okay not great but not terrible"

    default_text = st.session_state.get("example", "")

    # ===================== TEXT INPUT =====================
    review = st.text_area(
        "✍️ Enter a movie review:",
        value=default_text,
        placeholder="Type your movie review here...",
        height=160
    )

    char_count = len(review)
    st.caption(f"📝 Characters: {char_count}")

    # ===================== PREDICTION =====================
    if st.button("🚀 Analyze Sentiment", use_container_width=True):

        if review.strip() == "":
            st.warning("Please enter some text.")
        else:
            processed, token_len = preprocess_text(review)
            prob = float(model.predict(processed)[0][0])
            percent = int(prob * 100)

            # ===================== BAND LOGIC =====================
            if percent >= 70:
                band = "🟢 Strong Positive"
            elif percent >= 40:
                band = "🟡 Neutral / Uncertain"
            else:
                band = "🔴 Negative"

            # ===================== DISPLAY =====================
            st.markdown("---")
            st.subheader("📊 Prediction Result")

            st.progress(percent)
            st.metric("Sentiment Confidence", f"{percent}%")
            st.write(band)

            # ===================== MODEL LIMITATION =====================
            st.info("ℹ️ Simple RNN may struggle with negation and long sentences.")

            # ===================== EXPLANATION =====================
            st.subheader("🔍 Sentence Insight")
            for e in explain_sentence(review):
                st.write(e)

            # ===================== PREPROCESSING PREVIEW =====================
            with st.expander("🧠 See how text is processed"):
                st.write(f"🔢 Tokenized words: {token_len}")
                st.write(f"📏 Padded length: {MAX_LEN}")
                st.write("Text → Numbers → Padding → Embedding → RNN → Prediction")

            # ===================== SAVE HISTORY =====================
            st.session_state.history.insert(
                0,
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Text": review[:50] + "...",
                    "Confidence": percent,
                    "Result": band
                }
            )
            st.session_state.history = st.session_state.history[:5]

            # ===================== EXPORT =====================
            df = pd.DataFrame([{
                "Timestamp": datetime.now(),
                "Review": review,
                "Confidence (%)": percent,
                "Result": band
            }])

            st.download_button(
                "⬇️ Download Result (CSV)",
                df.to_csv(index=False),
                file_name="sentiment_result.csv",
                mime="text/csv"
            )

    # ===================== HISTORY =====================
    if st.session_state.history:
        st.subheader("🕒 Recent Predictions")
        st.table(pd.DataFrame(st.session_state.history))

    st.markdown("</div>", unsafe_allow_html=True)
