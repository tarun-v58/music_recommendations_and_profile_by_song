import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fuzzywuzzy import process
import numpy as np

# Load and pre-process data once at the start
def setup_recommender():
    try:
        data = pd.read_csv('spotify_dataset.csv')
        data.dropna(subset=['artists', 'track_name', 'track_genre'], inplace=True)
    except FileNotFoundError:
        return None, None, "Error: 'spotify_dataset.csv' not found. Please make sure the file is in the same directory."

    # Step 1: Data Preprocessing and Feature Combination
    data['combined_features'] = data['artists'].fillna('') + ' ' + \
                                data['track_name'].fillna('') + ' ' + \
                                data['track_genre'].fillna('')
    data['combined_features'] = data['combined_features'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x).lower())

    # Step 2: TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

    return data, tfidf_matrix, None

data, tfidf_matrix, error_message = setup_recommender()

# The existing recommendation function by song title
def get_recommendations_and_profile_by_song(song_title, df, tfidf_matrix, top_n=10):
    if error_message:
        return None, None, error_message, None, None
        
    song_titles = df['track_name'].tolist()
    best_match_tuple = process.extractOne(song_title, song_titles)
    
    if best_match_tuple and best_match_tuple[1] >= 85: 
        best_match_title = best_match_tuple[0]
        input_song_index = df[df['track_name'] == best_match_title].index[0]

        input_vector = tfidf_matrix[input_song_index].reshape(1, -1)
        cosine_sim_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

        sim_scores = list(enumerate(cosine_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
    
        song_indices = [i[0] for i in sim_scores]
        recommended_songs_df = df.iloc[song_indices][['track_name', 'artists', 'track_genre', 'popularity', 'danceability', 'energy', 'acousticness', 'valence']]

        taste_profile = {
            'avg_danceability': recommended_songs_df['danceability'].mean(),
            'avg_energy': recommended_songs_df['energy'].mean(),
            'avg_acousticness': recommended_songs_df['acousticness'].mean(),
            'avg_valence': recommended_songs_df['valence'].mean()
        }
        
        input_song_profile = df.iloc[input_song_index][['track_name', 'artists', 'track_genre', 'popularity', 'danceability', 'energy', 'acousticness', 'valence']].to_dict()

        return recommended_songs_df, input_song_profile, taste_profile, None, None
    else:
        return None, None, None, "Song not found or no close match.", None

# NEW FUNCTION: Recommend songs based on a single genre and sort by popularity
def get_recommendations_by_genre(genre, df, top_n=10):
    if error_message:
        return None, None
    
    # Use fillna('') before filtering to avoid errors on missing data
    filtered_df = df[
        (df['track_genre'].fillna('').str.contains(genre, case=False, na=False))
    ].sort_values(by='popularity', ascending=False).head(top_n)

    if not filtered_df.empty:
        return filtered_df[['track_name', 'artists', 'track_genre', 'popularity']].to_dict('records'), None
    else:
        return None, "No songs found for the selected genre. Please try a different option."