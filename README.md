# 🎬 MovieMatch — Movie Recommendation System

A content-based movie recommendation web application built
with Python, Streamlit, Pandas, and Scikit-learn.

MovieMatch helps users discover similar movies based on
movie genres and descriptions using machine learning techniques.

## Features

- Interactive movie selection
- Personalized content-based movie recommendations
- TF-IDF vectorization for text feature extraction
- Cosine similarity for movie comparison
- Adjustable number of recommendations
- Similarity scores displayed as percentages
- Interactive web interface built with Streamlit

## Technologies Used

- Python 3.13
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## How It Works

1. Movie genres and descriptions are combined into text features.
2. TF-IDF converts these text features into numerical vectors.
3. Cosine similarity measures how closely movies are related.
4. The application ranks movies by similarity.
5. The most similar movies are recommended to the user.

## Installation

Clone the repository:

    git clone https://github.com/esmkc0/Movie-Recommendation-System.git

Navigate to the project folder:

    cd Movie-Recommendation-System

Install the required libraries:

    python -m pip install -r requirements.txt

Run the application:

    python -m streamlit run main.py

Open http://localhost:8501 in your browser.

## Dataset

The application currently uses a curated sample dataset
of 30 movies with genres and short descriptions.

## Project Structure

    Movie-Recommendation-System/
    ├── main.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Concepts Demonstrated

- Content-Based Recommendation Systems
- Natural Language Processing (NLP)
- TF-IDF Feature Extraction
- Cosine Similarity
- Data Processing with Pandas
- Interactive Web Application Development

## Author

Developed as a Python software engineering portfolio project.