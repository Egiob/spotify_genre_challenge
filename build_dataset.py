import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd

LOCAL_PATH = 'data'
SPOTIPY_CLIENT_ID = 'd5a0f30e90834cccb74601b7211e2b1a'
SPOTIPY_CLIENT_SECRET = '5a10ab62b4614cfb971bb05b2bf8ae8b'


def process_playlist(playlist_uri, application):
    tracks_uri = []
    tracks_genres = []
    total_tracks = application.playlist(playlist_uri)['tracks']['total']
    
    for offset in range(0, total_tracks, 100):
        for track in application.playlist_items(playlist_id=playlist_uri, offset=offset)['items']:
            track_uri = track['track']['uri']
            artist_uri = track['track']['artists'][0]['uri']
            track_genres = application.artist(artist_uri)['genres']
            
            tracks_uri.append(track_uri)
            tracks_genres.append(track_genres)
    return tracks_uri, tracks_genres


def playlist_to_dataset(playlist_uri, application):
    tracks_uri, tracks_genres = process_playlist(playlist_uri, application)
    
    # audio analysis
    sections, segments = [], []
    
    # audio features
    danceability, energy = [], []
    key, loudness, mode = [], [], []
    speechiness, acousticness, instrumentalness, liveness, valence = [], [], [], [], []
    tempo, duration_ms, time_signature = [], [], []
    
    for i, uri in enumerate(tracks_uri):
        audio_analysis = application.audio_analysis(uri)
        audio_features = application.audio_features(uri)[0]
    
        sections.append(audio_analysis['sections'])
        segments.append(audio_analysis['segments'])
    
        #the lines commented correspond to duplicates
        danceability.append(audio_features['danceability'])
        energy.append(audio_features['energy'])
        key.append(audio_features['key'])
        loudness.append(audio_features['loudness'])
        mode.append(audio_features['mode'])
        speechiness.append(audio_features['speechiness'])
        acousticness.append(audio_features['acousticness'])
        instrumentalness.append(audio_features['instrumentalness'])
        liveness.append(audio_features['liveness'])
        valence.append(audio_features['valence'])
        tempo.append(audio_features['tempo'])
        duration_ms.append(audio_features['duration_ms'])
        time_signature.append(audio_features['time_signature'])
           
    data = pd.DataFrame({'track_uri': tracks_uri, 'track_genres': tracks_genres,
                         'danceability':danceability, 'energy':energy, 'speechiness':speechiness,
                         'acousticness':acousticness, 'instrumentalness':instrumentalness,
                         'liveness':liveness, 'valence':valence, 'tempo':tempo, 'mode':mode,
                         'duration_ms':duration_ms, 'time_signature':time_signature, 'key':key,
                         'sections':sections, 'segments':segments, 'loudness':loudness
                        })
        
    return data


def merge_playlists_data(playlists_uri, application):
    
    data = []
    for playlist_uri in playlists_uri:
        data.append(playlist_to_dataset(playlist_uri, application))
        print("One playlist done")
    
    dataset = pd.concat(data, axis=0, ignore_index=True)
    dataset = dataset.drop_duplicates(subset='track_uri').reset_index()
    
    return dataset


def build_dataset():
    
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                                              client_secret=SPOTIPY_CLIENT_SECRET))
    
    playlists_uri = ['6gS3HhOiI17QNojjPuPzqc',
                 '6s5MoZzR70Qef7x4bVxDO1'
                 '7dowgSWOmvdpwNkGFMUs6e',
                 '3pDxuMpz94eDs7WFqudTbZ',
                 '3HYK6ri0GkvRcM6GkKh0hJ',
                 '1IGB0Uz7x2VY28qMagUC24',
                 '0zJrEnj3O8CohOpFFUVSo9'
                ]
    
    dataset = merge_playlists_data(playlists_uri, spotify)
    dataset.to_csv('spotify_dataset.csv', sep=';')
    print("File saved in the Data directory")
    

if __name__ == "__main__":
    build_dataset()