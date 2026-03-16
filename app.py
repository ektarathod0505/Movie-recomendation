import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- Load Dataset --------
df = pd.read_csv("movies.csv")

# -------- Parse List Columns --------
def parse_names(val, top=5):
    try:
        items = ast.literal_eval(val)
        return [i['name'].replace(' ', '') for i in items[:top] if 'name' in i]
    except:
        return []

df['genres_clean']   = df['genres'].apply(parse_names)
df['keywords_clean'] = df['keywords'].apply(parse_names)
df['overview_clean'] = df['overview'].fillna('').apply(lambda x: x.split())

# -------- Build Tags --------
# genres (3x weight) + keywords (2x) + overview
# NO cast/crew — they add noise and hurt recommendations
df['tags'] = (
    df['genres_clean']   * 3 +
    df['keywords_clean'] * 2 +
    df['overview_clean']
)
df['tags'] = df['tags'].apply(lambda x: ' '.join(x).lower())

# -------- Prepare Final DataFrame --------
movies = df[['id', 'title', 'tags']].rename(columns={'id': 'movie_id'})
movies = movies.dropna(subset=['title'])
movies = movies[movies['tags'].str.strip() != ''].reset_index(drop=True)

print(f"Total movies: {len(movies)}")

# -------- Vectorize --------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])
similarity = cosine_similarity(vectors)

# -------- Test --------
def test_recommend(title):
    try:
        idx = movies[movies['title'] == title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        print(f"\nRecommendations for '{title}':")
        for i in distances[1:6]:
            print(f"  - {movies.iloc[i[0]]['title']}")
    except:
        print(f"Movie '{title}' not found")

test_recommend("Spider-Man 3")
test_recommend("Avatar")
test_recommend("The Dark Knight")

# -------- Save Model --------
os.makedirs("model", exist_ok=True)
pickle.dump(movies[['movie_id', 'title']], open("model/movie_list.pkl", "wb"))
pickle.dump(similarity, open("model/similarity.pkl", "wb"))

print("\nModel saved to model/ folder!")
