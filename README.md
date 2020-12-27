# spotify_genre_challenge
A machine learning challenge which consists in predicting the genre of a music based on features from the Spotify API.


### Labels
The labels are a vector of size (n_samples, n_genres), where each genre is attributed a column index in the labels array. For the i-th line (the i-th music) we put a 1 in the j-th column (the j-th genre) if the music belong to this genre. Since a music can have multiple genre, there is often several ones on a single line.


