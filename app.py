import pickle
import streamlit as st
import requests
import os

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

def recommend(movie):
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

    for i in distances[1:6]:  # top 5 recommendations
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters

# ---------------------- STREAMLIT APP ----------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.header('🎬 Movie Recommender System')

# Check if model files exist
if not os.path.exists('model/movie_list.pkl') or not os.path.exists('model/similarity.pkl'):
    st.error("Required model files are missing in the 'model/' folder. Please add 'movie_list.pkl' and 'similarity.pkl'.")
else:
    # Load models
    movies = pickle.load(open('model/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('model/similarity.pkl', 'rb'))

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

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
