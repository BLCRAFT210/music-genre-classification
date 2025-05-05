import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('spotify_songs.csv')

df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')

feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']

df_clean = df.dropna(subset=feature_cols + ['playlist_genre', 'playlist_subgenre'])

scaler = StandardScaler()
df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])


from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

df['genre_label'] = LabelEncoder().fit_transform(df['playlist_genre'])
df['subgenre_label'] = LabelEncoder().fit_transform(df['playlist_subgenre'])

X = df[feature_cols]
y_genre = df['genre_label']
y_subgenre = df['subgenre_label']

f_scores_genre, _ = f_classif(X, y_genre)
f_scores_subgenre, _ = f_classif(X, y_subgenre)

mi_scores_genre = mutual_info_classif(X, y_genre, discrete_features=False)
mi_scores_subgenre = mutual_info_classif(X, y_subgenre, discrete_features=False)

feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'ANOVA_F_Genre': f_scores_genre,
    'Mutual_Info_Genre': mi_scores_genre,
    'ANOVA_F_Subgenre': f_scores_subgenre,
    'Mutual_Info_Subgenre': mi_scores_subgenre
}).sort_values(by='Mutual_Info_Genre', ascending=False)

print(feature_importance_df)


df['genre_label'] = LabelEncoder().fit_transform(df['playlist_genre'])
df['subgenre_label'] = LabelEncoder().fit_transform(df['playlist_subgenre'])

X = df[feature_cols]
y_genre = df['genre_label']
y_subgenre = df['subgenre_label']

mi_genre = mutual_info_classif(X, y_genre, discrete_features=False, random_state=42)
mi_subgenre = mutual_info_classif(X, y_subgenre, discrete_features=False, random_state=42)

mi_genre_df = pd.DataFrame({'Feature': feature_cols, 'Mutual Information': mi_genre})
mi_genre_df = mi_genre_df.sort_values(by='Mutual Information', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Mutual Information', y='Feature', data=mi_genre_df, palette='coolwarm')
plt.title("Mutual Information of Features for Genre")
plt.xlabel("Mutual Information")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

mi_subgenre_df = pd.DataFrame({'Feature': feature_cols, 'Mutual Information': mi_subgenre})
mi_subgenre_df = mi_subgenre_df.sort_values(by='Mutual Information', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Mutual Information', y='Feature', data=mi_subgenre_df, palette='magma')
plt.title("Mutual Information of Features for Subgenre")
plt.xlabel("Mutual Information")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()