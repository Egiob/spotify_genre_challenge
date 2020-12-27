# Metric computation

## Dictionaries
The files
 - genre_index.dict
 - index_genre.dict
contain the serialized python dictionaries that map genre to corresponding target index (*resp.* target index to corresponding genre).
They can be loaded with the *joblib* library :
  ```python3
  import joblib
  joblib.load("genre_index.dict")
  ```
