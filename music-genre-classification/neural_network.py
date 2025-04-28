import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset

df = pd.read_csv('tracks.csv')  # Replace with your CSV file path

# Identify input features and label columns
feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms',
                'key_10', 'key_7', 'key_11', 'key_2', 'key_3', 'key_6',
                'key_1', 'key_4', 'key_0', 'key_8', 'key_5', 'key_9',
                'mode_0', 'mode_1']  # Added new features
label_cols = [col for col in df.columns if col.startswith('label_')]

# Convert one-hot label columns to a single categorical label
df['label'] = df[label_cols].idxmax(axis=1)

# Encode labels as integers
label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}
label_mapping = {idx: label.split('_')[1] for label, idx in label_to_index.items()}  # Map index to genre
df['label_index'] = df['label'].map(label_to_index)

# Split data
X = df[feature_cols].values
y = df['label_index'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

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

'''

# Move model to device
model = SongGenreClassifier(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 30  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in loss to qualify as an improvement
best_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(5):
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

    # Early stopping logic
    if total_loss < best_loss - min_delta:
        best_loss = total_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Set model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # Get model predictions
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)  # Get index of max log-probability

    # Convert to NumPy for comparison
    y_pred = predicted.cpu().numpy()  # Move to CPU before converting to NumPy
    y_true = y_test_tensor.cpu().numpy()  # Move to CPU before converting to NumPy

    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

'''

# Define a range of learning rates to test
learning_rates = [0.0001, 0.0003, 0.0005, 0.0008, 0.001]
results = {}
accuracies = {}
genre_accuracies = {}

for lr in learning_rates:
    print(f"\nTesting learning rate: {lr}")
    
    # Reinitialize the model and optimizer for each learning rate
    model = SongGenreClassifier(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train for 200 epochs
    losses = []
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
        
        losses.append(total_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    # Store the losses for this learning rate
    results[lr] = losses

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        y_pred = predicted.cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

        # Compute overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        accuracies[lr] = accuracy
        print(f"Test Accuracy with learning rate {lr}: {accuracy * 100:.2f}%")

        # Compute genre accuracy
        # Extract genres from labels (assumes labels are in the format 'label_genre_subgenre')

        y_pred_genres = [label_mapping[label] for label in y_pred]
        y_true_genres = [label_mapping[label] for label in y_true]

        # Compute genre accuracy
        genre_accuracy = accuracy_score(y_true_genres, y_pred_genres)
        print(f"Genre Accuracy with learning rate {lr}: {genre_accuracy * 100:.2f}%")
        genre_accuracies[lr] = genre_accuracy

# Print final test accuracies for each learning rate
print("\nFinal Test Accuracies:")
for lr, accuracy in accuracies.items():
    print(f"Learning Rate: {lr}, Test Accuracy: {accuracy * 100:.2f}%")

print("\nFinal Genre Accuracies:")
for lr, genre_accuracy in genre_accuracies.items():
    print(f"Learning Rate: {lr}, Genre Accuracy: {genre_accuracy * 100:.2f}%")