import requests

# Replace with your actual TMDb API key
TMDB_API_KEY = '69f5f834cdfaafe4db79e61d7fc6f9c3'


def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()
