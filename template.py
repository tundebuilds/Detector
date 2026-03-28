import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #333;
        }
        .subtitle {
            text-align: center;
            color: gray;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<div class="title">📰 Fake News Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check if a news article is Real or Fake using AI</div>', unsafe_allow_html=True)

# Input options
option = st.radio("Choose input method:", ["Type/Paste Text", "Upload .txt File"])

user_input = ""

if option == "Type/Paste Text":
    user_input = st.text_area("Paste your news content here:", height=200)

elif option == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("File Content:", user_input, height=200)

# Prediction
if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter or upload some text.")
    else:
        # Transform input
        input_data = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        fake_prob = probabilities[0][1]
        real_prob = probabilities[0][0]

        st.markdown("---")

        # Result display
        if prediction[0] == 1:
            st.error("🚨 This news is likely FAKE")
            st.progress(float(fake_prob))
            st.write(f"Confidence: **{fake_prob * 100:.2f}% Fake**")
        else:
            st.success("✅ This news is likely REAL")
            st.progress(float(real_prob))
            st.write(f"Confidence: **{real_prob * 100:.2f}% Real**")

        # Extra info
        with st.expander("🧠 How this works"):
            st.write("""
            This app uses a Machine Learning model trained on real and fake news datasets.
            
            - Text is converted into numerical features using a vectorizer
            - The model analyzes patterns in the text
            - It then predicts whether the news is real or fake
            
            ⚠️ Note: This is an AI prediction and may not always be 100% accurate.
            """)