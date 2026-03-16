import pickle
import streamlit as st
import requests
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- POSTER FUNCTION ----------------

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"

    try:
        data = requests.get(url).json()
        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return None
    except:
        return None


# ---------------- DATA LOADING ----------------

@st.cache_resource
def load_data():

    if os.path.exists("model/movie_list.pkl") and os.path.exists("model/similarity.pkl"):
        movies = pickle.load(open("model/movie_list.pkl","rb"))
        similarity = pickle.load(open("model/similarity.pkl","rb"))
        return movies, similarity

    elif os.path.exists("movies.csv"):

        movies = pd.read_csv("movies.csv")

        # Clean columns
        for col in ["genres","keywords","overview","cast","director"]:
            if col in movies.columns:
                movies[col] = movies[col].fillna("")

        # Create better tags
        movies["tags"] = (
            movies.get("genres","") + " " +
            movies.get("keywords","") + " " +
            movies.get("overview","") + " " +
            movies.get("cast","") + " " +
            movies.get("director","")
        )

        movies["tags"] = movies["tags"].str.lower()

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(
            max_features=15000,
            stop_words="english",
            ngram_range=(1,2)
        )

        vectors = tfidf.fit_transform(movies["tags"])

        similarity = cosine_similarity(vectors)

        os.makedirs("model", exist_ok=True)

        pickle.dump(movies, open("model/movie_list.pkl","wb"))
        pickle.dump(similarity, open("model/similarity.pkl","wb"))

        return movies, similarity

    else:
        return None, None


# ---------------- RECOMMEND FUNCTION ----------------

def recommend(movie, movies, similarity):

    try:
        index = movies[movies["title"] == movie].index[0]
    except:
        st.error("Movie not found")
        return [], []

    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )

    names = []
    posters = []

    for i in distances[1:6]:

        row = movies.iloc[i[0]]

        if "id" in movies.columns:
            movie_id = row["id"]
        elif "movie_id" in movies.columns:
            movie_id = row["movie_id"]
        elif "movieId" in movies.columns:
            movie_id = row["movieId"]
        else:
            movie_id = None

        names.append(row["title"])

        if movie_id:
            posters.append(fetch_poster(movie_id))
        else:
            posters.append(None)

    return names, posters


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

with st.spinner("Loading movies..."):
    movies, similarity = load_data()

if movies is None:

    st.error("Dataset not found.")
    st.info("Please upload movies.csv")

else:

    st.success(f"{len(movies)} movies loaded successfully!")

    movie_list = movies["title"].values

    selected_movie = st.selectbox(
        "Select a movie",
        movie_list
    )

    if st.button("Show Recommendation"):

        names, posters = recommend(selected_movie, movies, similarity)

        cols = st.columns(5)

        for i in range(5):

            with cols[i]:

                st.text(names[i])

                if posters[i]:
                    st.image(posters[i])
                else:
                    st.write("Poster not available")
