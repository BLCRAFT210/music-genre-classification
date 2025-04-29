import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df = pd.read_csv('tracks.csv')

feature_cols = [
    # Original features
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
    
    # Basic interactions
    'energy_dance', 'tempo_energy', 'acoustic_instrument', 'valence_energy',
    'duration_min', 'loudness_per_min',
    
    # Log transforms
    'log_duration', 'log_loudness', 'log_instrumentalness',
    
    # Ratios and differences
    'energy_loudness_ratio', 'valence_minus_energy', 'dance_tempo_ratio',
    'acoustic_instrument_ratio', 'speech_to_instrument_ratio',
    
    # Polynomial features
    'energy_squared', 'tempo_squared', 'valence_squared', 'danceability_squared',
    
    # Complex interactions
    'dance_energy_tempo', 'valence_energy_loudness', 'acoustic_speech_live',
    
    # Key and mode one-hot features
    'key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 
    'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11',
    'mode_0', 'mode_1'
]

features = df[feature_cols]

# Create labels for main genres only and genre+subgenre
labels_main_only = df[[col for col in df.columns if col.startswith('label_')]].idxmax(axis=1).apply(lambda x: x.replace('label_', '').split('_')[0])
labels_main_and_sub = df[[col for col in df.columns if col.startswith('label_')]].idxmax(axis=1).apply(lambda x: x.replace('label_', ''))

print("Number of features:", len(feature_cols))

# First, attempting KNN with main genre labels
print("\nTraining KNN with main genres only...")
X_train, X_test, y_train, y_test = train_test_split(features, labels_main_only, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Main Genre Classification Results:")
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("\nClassification Report for main genres:\n", classification_report(y_test, y_pred))

# Now with genre+subgenre labels
print("\nTraining KNN with genre+subgenre...")
X_train, X_test, y_train, y_test = train_test_split(features, labels_main_and_sub, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Genre+Subgenre Classification Results:")
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("\nClassification Report for genre+subgenre:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning for main genres
print("\nTuning hyperparameters for main genres...")
X_train, X_test, y_train, y_test = train_test_split(features, labels_main_only, test_size=0.2, random_state=42)

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

print("\nTuned Model Results (Main Genres):")
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
