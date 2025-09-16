import pandas as pd
from flask import Flask, render_template, request
from recommender import get_recommendations_and_profile_by_song, get_recommendations_by_genre, data, tfidf_matrix, error_message

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations_by_song = None
    input_song_profile = None
    taste_profile = None
    message_song = None
    input_song = None
    
    recommendations_by_genre = None
    message_genre = None
    
    # Get unique artists and genres for dropdowns
    if data is not None:
        unique_genres = sorted(data['track_genre'].dropna().unique())
    else:
        unique_genres = []

    if request.method == 'POST':
        form_type = request.form.get('form_type')
        
        if form_type == 'song_form':
            input_song = request.form.get('song_title')
            if input_song and not error_message:
                recommendations_df, input_song_profile, taste_profile, message_song, _ = get_recommendations_and_profile_by_song(input_song, data, tfidf_matrix)
                
                if isinstance(recommendations_df, pd.DataFrame):
                    recommendations_by_song = recommendations_df.to_dict('records')
                
        elif form_type == 'genre_form':
            selected_genre = request.form.get('genre_select')
            if selected_genre:
                recommendations_by_genre, message_genre = get_recommendations_by_genre(selected_genre, data)

    return render_template('index.html', 
                           recommendations_by_song=recommendations_by_song,
                           input_song_profile=input_song_profile,
                           taste_profile=taste_profile,
                           message_song=message_song,
                           input_song=input_song,
                           recommendations_by_genre=recommendations_by_genre,
                           message_genre=message_genre,
                           unique_genres=unique_genres,
                           error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)