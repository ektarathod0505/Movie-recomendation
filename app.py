import streamlit as st
import requests
import os
import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "movies.csv")
    df = pd.read_csv(csv_path)

    def parse_names(val, top=5):
        try:
            items = ast.literal_eval(val)
            return [i['name'].replace(' ', '') for i in items[:top] if 'name' in i]
        except:
            return []

    df['genres_clean']   = df['genres'].apply(parse_names)
    df['keywords_clean'] = df['keywords'].apply(parse_names)
    df['overview_clean'] = df['overview'].fillna('').apply(lambda x: x.split())

    df['tags'] = (
        df['genres_clean']   * 3 +
        df['keywords_clean'] * 2 +
        df['overview_clean']
    )
    df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())

    movies = df[['id', 'title', 'tags']].rename(columns={'id': 'movie_id'})
    movies = movies.dropna(subset=['title'])
    movies = movies[movies['tags'].str.strip() != ''].reset_index(drop=True)

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
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    names, posters = [], []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))
    return names, posters

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
