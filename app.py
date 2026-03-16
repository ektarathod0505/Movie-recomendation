import pickle
import streamlit as st
import requests
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- POSTER FUNCTION ----------------------

def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return None

    except requests.exceptions.RequestException:
        return None


# ---------------------- DATA LOADING ----------------------

@st.cache_resource
def load_data():

    # Load saved model
    if os.path.exists("model/movie_list.pkl") and os.path.exists("model/similarity.pkl"):
        movies = pickle.load(open("model/movie_list.pkl", "rb"))
        similarity = pickle.load(open("model/similarity.pkl", "rb"))
        return movies, similarity

    # Build model from CSV
    elif os.path.exists("movies.csv"):

        movies = pd.read_csv("movies.csv")

        # Clean text columns
        features = []

        for col in ["genres", "keywords", "overview", "cast", "director"]:
            if col in movies.columns:
                movies[col] = movies[col].fillna("")
                features.append(col)

        # Create tags
        if features:
            movies["tags"] = movies[features].apply(
                lambda row: " ".join(row.values.astype(str)), axis=1
            )
        else:
            movies["tags"] = movies["title"]

        movies["tags"] = movies["tags"].str.lower()

        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1,2)
        )

        vectors = tfidf.fit_transform(movies["tags"]).toarray()

        similarity = cosine_similarity(vectors)

        os.makedirs("model", exist_ok=True)

        pickle.dump(movies, open("model/movie_list.pkl", "wb"))
        pickle.dump(similarity, open("model/similarity.pkl", "wb"))

        return movies, similarity

    else:
        return None, None


# ---------------------- RECOMMEND FUNCTION ----------------------

def recommend(movie, movies, similarity):

    try:
        index = movies[movies["title"] == movie].index[0]
    except IndexError:
        st.error("Movie not found")
        return [], []

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []
    recommended_posters = []

    for i in distances[1:6]:

        movie_row = movies.iloc[i[0]]

        # Detect ID column
        if "movie_id" in movies.columns:
            movie_id = movie_row["movie_id"]
        elif "id" in movies.columns:
            movie_id = movie_row["id"]
        elif "movieId" in movies.columns:
            movie_id = movie_row["movieId"]
        else:
            movie_id = None

        recommended_movies.append(movie_row["title"])

        if movie_id:
            poster = fetch_poster(movie_id)
        else:
            poster = None

        recommended_posters.append(poster)

    return recommended_movies, recommended_posters


# ---------------------- STREAMLIT APP ----------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

with st.spinner("Loading movie dataset..."):
    movies, similarity = load_data()

if movies is None:

    st.error("Dataset not found.")
    st.info("Upload movies.csv or model/movie_list.pkl")

else:

    st.success(f"Loaded {len(movies)} movies successfully!")

    movie_list = movies["title"].values

    selected_movie = st.selectbox(
        "Select a movie",
        movie_list
    )

    if st.button("Show Recommendation"):

        names, posters = recommend(selected_movie, movies, similarity)

        if names:

            cols = st.columns(5)

            for i in range(5):

                with cols[i]:

                    st.text(names[i])

                    if posters[i]:
                        st.image(posters[i])
                    else:
                        st.write("Poster not available")

        else:
            st.warning("No recommendations found.")
