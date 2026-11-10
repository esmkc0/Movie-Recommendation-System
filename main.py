
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="MovieMatch",
    page_icon="🎬",
    layout="wide"
)

MOVIES = [
    ("Inception", "Sci-Fi Thriller", "A thief enters dreams to steal secrets and explores reality."),
    ("Interstellar", "Sci-Fi Adventure", "Astronauts travel through space to save humanity."),
    ("The Matrix", "Sci-Fi Action", "A hacker discovers a simulated reality and fights machines."),
    ("The Dark Knight", "Action Crime", "A superhero battles a criminal mastermind in Gotham."),
    ("Batman Begins", "Action Crime", "A billionaire becomes a masked hero fighting crime."),
    ("The Batman", "Action Crime Mystery", "A detective superhero investigates murders in Gotham."),
    ("Avengers Endgame", "Action Superhero Sci-Fi", "Superheroes unite to reverse a cosmic disaster."),
    ("Iron Man", "Action Superhero Sci-Fi", "An inventor builds an advanced armored superhero suit."),
    ("Spider-Man Homecoming", "Action Superhero Comedy", "A teenage superhero balances school and crime fighting."),
    ("Guardians of the Galaxy", "Sci-Fi Action Comedy", "Space adventurers team up to protect the galaxy."),
    ("Dune", "Sci-Fi Adventure Drama", "A young noble travels to a desert planet with political conflict."),
    ("Star Wars A New Hope", "Sci-Fi Adventure", "A young hero joins rebels in a galactic battle."),
    ("Arrival", "Sci-Fi Drama Mystery", "A linguist communicates with mysterious alien visitors."),
    ("The Martian", "Sci-Fi Adventure", "An astronaut survives alone on Mars using science."),
    ("Gravity", "Sci-Fi Thriller", "An astronaut struggles to survive after a space disaster."),
    ("Titanic", "Romance Drama", "Two strangers fall in love aboard a doomed ocean liner."),
    ("The Notebook", "Romance Drama", "A couple experiences love and separation across decades."),
    ("La La Land", "Romance Musical Drama", "A musician and an actress pursue dreams and love."),
    ("Pride and Prejudice", "Romance Drama", "Two people overcome prejudice and social expectations."),
    ("The Fault in Our Stars", "Romance Drama", "Two teenagers form a meaningful romantic relationship."),
    ("The Conjuring", "Horror Mystery", "Paranormal investigators confront a terrifying haunting."),
    ("Insidious", "Horror Thriller", "A family experiences supernatural events in their home."),
    ("A Quiet Place", "Horror Sci-Fi Thriller", "A family survives monsters that hunt using sound."),
    ("Get Out", "Horror Mystery Thriller", "A young man discovers disturbing secrets during a visit."),
    ("It", "Horror Mystery", "Children confront a terrifying supernatural creature."),
    ("Toy Story", "Animation Comedy Adventure", "Living toys experience friendship and adventure."),
    ("Finding Nemo", "Animation Adventure Family", "A father searches the ocean for his missing son."),
    ("Inside Out", "Animation Comedy Family", "Emotions guide a young girl through life changes."),
    ("Coco", "Animation Family Fantasy", "A boy explores the world of the dead and family memories."),
    ("Shrek", "Animation Comedy Fantasy", "An ogre begins a funny adventure to rescue a princess.")
]


@st.cache_data
def load_movies():
    return pd.DataFrame(
        MOVIES,
        columns=["title", "genres", "description"]
    )


@st.cache_resource
def build_model():
    movies = load_movies()

    combined_text = (
        movies["genres"] + " " +
        movies["genres"] + " " +
        movies["description"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(combined_text)
    return cosine_similarity(matrix)


def recommend_movies(title, number=5):
    movies = load_movies()
    similarity = build_model()

    movie_index = movies.index[
        movies["title"] == title
    ][0]

    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(
        scores,
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []

    for index, score in scores:
        if index != movie_index:
            recommendations.append({
                "title": movies.iloc[index]["title"],
                "genres": movies.iloc[index]["genres"],
                "description": movies.iloc[index]["description"],
                "similarity": score
            })

        if len(recommendations) == number:
            break

    return recommendations


st.title("🎬 MovieMatch")
st.subheader("Discover your next favorite movie!")

st.write(
    "Choose a movie you enjoy and discover similar "
    "movies using machine learning."
)

movies = load_movies()

with st.sidebar:
    st.header("🎥 Movie Explorer")

    st.metric("Movies Available", len(movies))

    st.markdown("---")

    st.write("**Recommendation Algorithm**")
    st.write("TF-IDF Vectorization")
    st.write("Cosine Similarity")

    st.markdown("---")

    st.caption("Built with Python, Streamlit and Scikit-learn")

selected_movie = st.selectbox(
    "🍿 Select a movie:",
    movies["title"].tolist()
)

number_of_recommendations = st.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("✨ Find Similar Movies", type="primary"):
    recommendations = recommend_movies(
        selected_movie,
        number_of_recommendations
    )

    st.success(
        f"Here are movies similar to {selected_movie}!"
    )

    for movie in recommendations:
        with st.container(border=True):
            st.subheader("🎞️ " + movie["title"])
            st.write("**Genres:** " + movie["genres"])
            st.write(movie["description"])

            st.progress(
                min(max(float(movie["similarity"]), 0.0), 1.0),
                text=f"Similarity: {movie['similarity'] * 100:.1f}%"
            )

st.markdown("---")
st.caption(
    "MovieMatch | Content-Based Movie Recommendation System"
)
