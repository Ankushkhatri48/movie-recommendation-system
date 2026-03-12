import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load movie data
movies = pickle.load(open('movies.pkl','rb'))

# -------- CREATE SIMILARITY MATRIX --------
@st.cache_data
def create_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = create_similarity()

# -------- POSTER FUNCTION --------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=5457163f23182cf954b7007eb965982e&language=en-US"
        data = requests.get(url).json()

        if data.get("poster_path") is None:
            return "https://via.placeholder.com/300x450?text=No+Poster"

        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

    except:
        return "https://via.placeholder.com/300x450?text=Poster+Error"

# -------- GENRE LIST --------
def get_genre_list():
    genres = set()
    for g in movies['genres']:
        for item in g:
            genres.add(item)
    return sorted(list(genres))

genre_list = get_genre_list()

# -------- UI --------
st.title("🎬 Movie Recommendation System")

selected_genre = st.selectbox(
    "Select Genre",
    ["All"] + genre_list
)

# -------- FILTER MOVIES --------
if selected_genre == "All":
    filtered_movies = movies
else:
    filtered_movies = movies[movies['genres'].apply(lambda x: selected_genre in x)]

movie_list = filtered_movies['title'].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

# -------- RECOMMEND FUNCTION --------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# -------- BUTTON --------
if st.button("Recommend"):

    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.caption(names[i])