# spotify_genre_challenge
A machine learning challenge which consists in predicting the genre of a music based on features from the Spotify API.


### Labels
The labels are a vector of size (n_samples, n_genres), where each genre is attributed a column index in the labels array. For the i-th line (the i-th music) we put a 1 in the j-th column (the j-th genre) if the music belongs to this genre. Since a music can have multiple genres, there are often several 1's on a single line.

### Metrics evaluation
We have used a single metric : the Earth Mover's Distance (see [https://fr.wikipedia.org/wiki/Earth_mover%27s_distance](https://fr.wikipedia.org/wiki/Earth_mover%27s_distance)), which gives a measure of how close the genre we have predicted is close to the true genre. For example the genre *hip-hop* is quite close to the genre *rap* but quite far to the genre *pop*. To define the ground metric, we use a measure of similarity between each genre. 
