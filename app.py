import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )


emotion_classifier = load_model()

st.title("Words That Find You")

st.write("Tell me what's on your mind.")

text = st.text_area("How are you feeling?")

if st.button("Find My Words"):
    if text:
        result = emotion_classifier(text)[0]

        emotion = result["label"]
        confidence = result["score"]

        st.subheader("I think you're feeling...")

        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Confidence:** {confidence:.2%}")

    else:
        st.warning("Please tell me what's on your mind first.")