import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title
st.title("📰 Fake News Detection App")

st.write("Enter a news article below to check if it's **Real or Fake**.")

# Input box
user_input = st.text_area("Paste your news content here:")

# Prediction button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        input_data = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_data)

        # Output result
        if prediction[0] == 1:
            st.error("🚨 This news is likely FAKE.")
        else:
            st.success("✅ This news is likely REAL.")