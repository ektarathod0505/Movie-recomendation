import streamlit as st
import requests
import os
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "movies.csv")
    df = pd.read_csv(csv_path)

    def parse_names(val, top=8):
        try:
            items = ast.literal_eval(val)
            return [i['name'].replace(' ', '') for i in items[:top] if 'name' in i]
        except:
            return []

    df['genres_clean']   = df['genres'].apply(lambda x: parse_names(x, top=5))
    df['keywords_clean'] = df['keywords'].apply(lambda x: parse_names(x, top=8))
    df['overview_clean'] = df['overview'].fillna('').apply(lambda x: x.split())

    # genres 4x, keywords 3x, overview 1x
    df['tags'] = (
        df['genres_clean']   * 4 +
        df['keywords_clean'] * 3 +
        df['overview_clean'] * 1
    )
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())

    movies = df[['id', 'title', 'tags', 'vote_average', 'vote_count']].rename(columns={'id': 'movie_id'})
    movies = movies.dropna(subset=['title'])
    movies = movies[movies['tags'].str.strip() != ''].reset_index(drop=True)

    # Popularity score (Bayesian weighted)
    movies['vote_count']   = pd.to_numeric(movies['vote_count'],   errors='coerce').fillna(0)
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)
    movies['pop_score']    = (movies['vote_average'] * movies['vote_count']) / (movies['vote_count'] + 500)
    movies['pop_norm']     = movies['pop_score'] / movies['pop_score'].max()

    # Content similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags'])
    similarity = cosine_similarity(vectors)

    return movies, similarity

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]

    # Blend 85% content similarity + 15% popularity
    sim_scores  = similarity[index].copy()
    pop_scores  = movies['pop_norm'].values
    final_scores = 0.85 * sim_scores + 0.15 * pop_scores
    final_scores[index] = 0  # exclude self

    top5 = np.argsort(final_scores)[::-1][:5]

    names, posters = [], []
    for i in top5:
        names.append(movies.iloc[i]['title'])
        posters.append(fetch_poster(movies.iloc[i]['movie_id']))
    return names, posters

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")

with st.spinner("Loading movies..."):
    movies, similarity = load_data()

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button("Show Recommendation"):
    with st.spinner("Finding similar movies..."):
        names, posters = recommend(selected_movie, movies, similarity)
    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            st.image(poster)
