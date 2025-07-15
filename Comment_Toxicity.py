
import nltk
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# NLTK resource download
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and preprocess data
path = r"D:/GUVI/MAE3/Data/train.csv"
df = pd.read_csv(path)
df['clean_comment'] = df['comment_text'].apply(preprocess)
df['label'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].max(axis=1)

X = df['clean_comment']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model Definitions
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

class ToxicCNN(nn.Module):
    def __init__(self, input_dim):
        super(ToxicCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 100, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(100 * ((input_dim - 5 + 1) // 2), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.sigmoid(self.fc(x))

class ToxicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ToxicLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x.long())
        _, (h_n, _) = self.lstm(x)
        return self.sigmoid(self.fc(h_n[-1]))

# Model selection and training
model_type = 'mlp'  # change to 'cnn' or 'lstm'
model = None
if model_type == 'mlp':
    model = ToxicClassifier(X_train_vec.shape[1])
elif model_type == 'cnn':
    model = ToxicCNN(X_train_vec.shape[1])
elif model_type == 'lstm':
    model = ToxicLSTM(X_train_vec.shape[1])
else:
    raise ValueError("Invalid model type")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), f"toxic_{model_type}_model.pth")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).squeeze()
    predictions = torch.round(predictions)

accuracy = accuracy_score(y_test_tensor, predictions)
precision = precision_score(y_test_tensor, predictions)
recall = recall_score(y_test_tensor, predictions)
f1Score = f1_score(y_test_tensor, predictions)

print("\n=== Evaluation Results ===")
print(f"Accuracy:  {accuracy}")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1 Score:  {f1Score}")
