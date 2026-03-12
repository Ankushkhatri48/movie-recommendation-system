import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
h1 {
    color: #E50914;
}
.movie-title {
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
movies = pickle.load(open('movies.pkl','rb'))

# ---------------- CREATE SIMILARITY ----------------
@st.cache_data
def create_similarity():
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = create_similarity()

# ---------------- FETCH MOVIE DETAILS ----------------
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=5457163f23182cf954b7007eb965982e&language=en-US"
        data = requests.get(url).json()

        poster = "https://image.tmdb.org/t/p/w500/" + data["poster_path"] if data.get("poster_path") else "https://via.placeholder.com/300x450"
        rating = data.get("vote_average", "N/A")
        overview = data.get("overview", "No description available")

        return poster, rating, overview

    except:
        return "https://via.placeholder.com/300x450", "N/A", "Error loading description"

# ---------------- GENRE LIST ----------------
def get_genre_list():
    genres = set()
    for g in movies['genres']:
        for item in g:
            genres.add(item)
    return sorted(list(genres))

genre_list = get_genre_list()

# ---------------- HEADER ----------------
st.title("🎬 Movie Recommendation System")
st.markdown("Find movies similar to your favorites instantly.")

# ---------------- TRENDING MOVIES ----------------
st.markdown("### 🔥 Trending Movies")

trending = movies.sample(5)
cols = st.columns(5)

for i, col in enumerate(cols):
    with col:
        poster, rating, _ = fetch_movie_details(trending.iloc[i].movie_id)
        st.image(poster, use_container_width=True)
        st.caption(trending.iloc[i].title)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎬 Movie Filters")

selected_genre = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + genre_list
)

# ---------------- FILTER MOVIES ----------------
if selected_genre == "All":
    filtered_movies = movies
else:
    filtered_movies = movies[movies['genres'].apply(lambda x: selected_genre in x)]

movie_list = filtered_movies['title'].values

selected_movie = st.sidebar.selectbox(
    "Select a movie",
    movie_list
)

recommend_button = st.sidebar.button("Recommend")

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    posters = []
    ratings = []
    overviews = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        poster, rating, overview = fetch_movie_details(movie_id)

        names.append(movies.iloc[i[0]].title)
        posters.append(poster)
        ratings.append(rating)
        overviews.append(overview)

    return names, posters, ratings, overviews

# ---------------- SHOW RESULTS ----------------
if recommend_button:

    with st.spinner("Finding best movies for you..."):

        names, posters, ratings, overviews = recommend(selected_movie)

    st.subheader("🎬 Recommended Movies")

    cols = st.columns(5)

    for i in range(5):

        with cols[i]:

            st.image(posters[i], use_container_width=True)

            st.markdown(f"<p class='movie-title'>{names[i]}</p>", unsafe_allow_html=True)

            st.write(f"⭐ Rating: {ratings[i]}")

            st.caption(overviews[i][:120] + "...")