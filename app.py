import pickle
import streamlit as st
import requests
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# ---------------------- FUNCTIONS ----------------------
 
def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return None
    except requests.exceptions.RequestException:
        return None
 
 
@st.cache_resource
def load_data():
    """Load movies and generate similarity matrix automatically."""
    
    # ── Option 1: Load from pickle if available ──
    if os.path.exists('model/movie_list.pkl') and os.path.exists('model/similarity.pkl'):
        movies = pickle.load(open('model/movie_list.pkl', 'rb'))
        similarity = pickle.load(open('model/similarity.pkl', 'rb'))
        return movies, similarity
 
    # ── Option 2: Generate from movies.csv ──
    elif os.path.exists('movies.csv'):
        movies = pd.read_csv('movies.csv')
 
        # Generate tags if not present
        if 'tags' not in movies.columns:
            # combine available text columns
            text_cols = []
            for col in ['genres', 'keywords', 'overview', 'cast', 'crew']:
                if col in movies.columns:
                    movies[col] = movies[col].fillna('')
                    text_cols.append(col)
            movies['tags'] = movies[text_cols].apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1
            )
 
        movies['tags'] = movies['tags'].fillna('').str.lower()
 
        # Build similarity matrix
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        similarity = cosine_similarity(vectors)
 
        # Save for next run
        os.makedirs('model', exist_ok=True)
        pickle.dump(movies, open('model/movie_list.pkl', 'wb'))
        pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
 
        return movies, similarity
 
    # ── Option 3: Load only movie_list.pkl ──
    elif os.path.exists('model/movie_list.pkl'):
        movies = pickle.load(open('model/movie_list.pkl', 'rb'))
 
        # Regenerate similarity from movie_list
        if 'tags' not in movies.columns:
            text_cols = []
            for col in ['genres', 'keywords', 'overview', 'cast', 'crew']:
                if col in movies.columns:
                    movies[col] = movies[col].fillna('')
                    text_cols.append(col)
            if text_cols:
                movies['tags'] = movies[text_cols].apply(
                    lambda row: ' '.join(row.values.astype(str)), axis=1
                )
            else:
                movies['tags'] = movies.get('title', '')
 
        movies['tags'] = movies['tags'].fillna('').str.lower()
 
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies['tags']).toarray()
        similarity = cosine_similarity(vectors)
 
        return movies, similarity
 
    else:
        return None, None
 
 
def recommend(movie, movies, similarity):
    """Return top 5 recommended movies and their posters."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in database.")
        return [], []
 
    distances = sorted(
        list(enumerate(similarity[index])),
        key=lambda x: x[1],
        reverse=True
    )
 
    recommended_movie_names = []
    recommended_movie_posters = []
 
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
 
    return recommended_movie_names, recommended_movie_posters
 
 
# ---------------------- STREAMLIT APP ----------------------
 
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.header('🎬 Movie Recommender System')
 
# Load data with spinner
with st.spinner('Loading movie data... please wait ⏳'):
    movies, similarity = load_data()
 
if movies is None:
    st.error("No data files found! Please make sure movies.csv or model/movie_list.pkl exists.")
    st.info("Required files: movies.csv OR model/movie_list.pkl")
 
else:
    st.success(f"✅ Loaded {len(movies)} movies successfully!")
 
    movie_list = movies['title'].values
 
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )
 
    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters = recommend(
            selected_movie, movies, similarity
        )
 
        if recommended_movie_names:
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.text(recommended_movie_names[i])
                    if recommended_movie_posters[i]:
                        st.image(recommended_movie_posters[i])
                    else:
                        st.write("Poster not available")
        else:
            st.warning("No recommendations found.")
