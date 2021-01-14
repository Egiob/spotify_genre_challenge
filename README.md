# spotify_genre_challenge
Contributors :
 - Ulysse Demay
 - Nadir Abdou
 - Mickael Corroyer
 - Nathan Xerri
 - Raphaël Boige
 
A machine learning challenge which consists in predicting the genre of a music based on features from the Spotify API.


### Labels
The labels are an array of size (n_samples, n_genres), where each genre is attributed a column index in the labels array. For the i-th line (the i-th music) we put a 1 in the j-th column (the j-th genre) if the music belongs to this genre. Since a music can have multiple genres, there are often several 1's on a single line.

### Metrics evaluation
We have used two metrics, one is the Earth Mover's Distance (see [https://en.wikipedia.org/wiki/Earth_mover%27s_distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)), which gives a measure of how close the genre we have predicted is close to the true genre. For example the genre *hip-hop* is quite close to the genre *rap* but quite far to the genre *pop*. To define the ground metric, we use a measure of similarity between each genre. 

### Dataset
To conceive the dataset we use the Spotify API and the playlists of the website [http://everynoise.com/](http://everynoise.com/). We sequentially add each track of the chosen playlists and we associate each of them with the genres of their artist. In addition we add the audio features of each track and the some features (segments and sections) of the audio analysis provided by Spotify.
In order to build the dataset some steps are needed:
- Open the file `data_constructor.ipynb` and fill the Spotify credentials with yours
- Execute the entire notebook and wait until it is over (it can take one hour or more)
- The dataset `spotify_dataset.csv` should have appeared in your directory

If you directly want to download this dataset you can use this link: [https://lufi.rezel.net/r/pTJyWBE1xQ#N7PeYCNts9W6u1Z0gMtFo5OpaeOe5RfKaf3v0bwr0Lk=](https://lufi.rezel.net/r/pTJyWBE1xQ#N7PeYCNts9W6u1Z0gMtFo5OpaeOe5RfKaf3v0bwr0Lk=).
