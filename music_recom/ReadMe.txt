Music Recommendation System
------------------------------------
------------------------------------
This project is a web-based music recommendation system built with Python, Flask, and scikit-learn. It offers two different methods for users to discover music:

Content-Based Recommendation:
 	Find songs similar to a favorite song you input.

Genre-Based Recommendation:
	Discover popular songs from a specific genre.

Features:

* Content-Based Filtering: The system analyzes a song's title, artist, and genre to recommend other songs with similar characteristics.

* Genre-Based Filtering: Users can select a genre and receive a list of the most popular songs from that category.

* User Taste Profile: The application analyzes the audio features (e.g., danceability, energy, and acousticness) of the recommended songs to give users insights into their musical taste.

Interactive Web Interface: A simple and intuitive frontend allows users to easily search for songs or browse by genre.

Robust Data Handling: The backend is designed to handle large datasets efficiently, using on-demand similarity calculations to avoid memory errors.

How It Works : 

The system's core logic is a content-based filtering engine.

Data Processing: The backend loads a music dataset and combines key features (track_name, artists, track_genre) into a single text string for each song.

TF-IDF Vectorization: The combined features are transformed into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This technique identifies the most important words in each song's description.

Cosine Similarity: When a user searches for a song, the system calculates the cosine similarity between the user's song vector and every other song vector in the dataset. This score measures how similar the songs are.

Recommendations: The system returns the top songs with the highest similarity scores. For genre-based recommendations, it simply filters by genre and sorts by the song's popularity.

Project Structure :

app.py: The main Flask application file. It handles routing, user input, and rendering the web pages.

recommender.py: Contains all the data processing and recommendation logic, including the TF-IDF vectorization and similarity calculations.

spotify_dataset.csv: The dataset containing music information (e.g., song title, artist, genre, popularity, and audio features).

templates/index.html: The frontend HTML file that provides the user interface for the application.