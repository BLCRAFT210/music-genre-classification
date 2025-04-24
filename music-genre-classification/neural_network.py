import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset

df = pd.read_csv('tracks.csv')  # Replace with your CSV file path

# Identify input features and label columns
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms']
label_cols = [col for col in df.columns if col.startswith('label_')]

# Convert one-hot label columns to a single categorical label
df['label'] = df[label_cols].idxmax(axis=1)

# Encode labels as integers
label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label_index'] = df['label'].map(label_to_index)

# Split data
X = df[feature_cols].values
y = df['label_index'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create dataset and dataloader
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Neural network
class SongGenreClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SongGenreClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(label_to_index)

model = SongGenreClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

from sklearn.metrics import accuracy_score

# Set model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # Get model predictions
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)  # Get index of max log-probability

    # Convert to NumPy for comparison
    y_pred = predicted.numpy()
    y_true = y_test_tensor.numpy()

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")