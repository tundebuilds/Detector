import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 1️⃣ Load separate files
# -------------------------------

with open("data/True.csv", "r", encoding="utf-8") as f:
    real_texts = f.readlines()

with open("data/Fake.csv", "r", encoding="utf-8") as f:
    fake_texts = f.readlines()

# -------------------------------
# 2️⃣ Create dataframes with labels
# -------------------------------

df_real = pd.DataFrame({"text": real_texts, "label": "real"})
df_fake = pd.DataFrame({"text": fake_texts, "label": "fake"})

# Merge
df = pd.concat([df_real, df_fake], ignore_index=True)

# -------------------------------
# 3️⃣ Clean & shuffle
# -------------------------------

df = df.dropna(subset=['text'])
df['label'] = df['label'].str.lower()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after cleaning:")
print(df['label'].value_counts())

# -------------------------------
# 4️⃣ Prepare features & labels
# -------------------------------

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1,2),
    min_df=2
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5️⃣ Train classifier
# -------------------------------

model = SGDClassifier(
    loss='hinge',
    penalty='l2',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 6️⃣ Evaluate model
# -------------------------------

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['fake', 'real']))
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=['fake', 'real']))

# -------------------------------
# 7️⃣ Save model & vectorizer
# -------------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved successfully!")

import subprocess
import sys

print("Launching Fake News App...")
subprocess.run([sys.executable, "src/fake_news_app.py"], check=True)