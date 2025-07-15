# toxic_streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load model artifacts
vectorizer = CountVectorizer(max_features=5000)
vectorizer.fit(pd.read_csv(r"D:/GUVI/MAE3/Data/train.csv")['comment_text'].apply(lambda x: re.sub(r'[^a-z\s]', '', x.lower())))

class ToxicClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ToxicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# Load trained model
model_type = 'mlp'
model = ToxicClassifier(5000)
model.load_state_dict(torch.load(f"toxic_{model_type}_model.pth", map_location=torch.device('cpu')))
model.eval()

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit App
st.title("Toxic Comment Classifier")
st.write("Enter a comment below to check if it's toxic:")

input_text = st.text_area("Comment Text", "")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess(input_text)
        vec = vectorizer.transform([cleaned_text]).toarray()
        input_tensor = torch.tensor(vec, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor).item()
            prediction = "Toxic" if output >= 0.5 else "Non-Toxic"
            st.success(f"Prediction: {prediction} (Confidence: {output:.2f})")
