# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:16:49 2023
@author: DELL
"""
import io
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from streamlit import set_page_config
import textract
import tempfile
import os
import pandas as pd

# Load the movie recommendation model and CountVectorizer
movie_model = pickle.load(open(r"C:\Users\DELL\Desktop\P300\p300_movie\deployment\movie_recommendation_model.pkl", "rb"))
count_vectorizer = pickle.load(open(r"C:\Users\DELL\Desktop\P300\p300_movie\deployment\count_vectorizer.pkl", "rb"))
filledna=pd.read_csv(r'C:\Users\DELL\Desktop\P300\p300_movie\deployment\filledna.csv')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and non-alphanumeric characters
    #text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    processed_text = ' '.join(words)
    return processed_text

def vectorize_text(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Transform the preprocessed text using CountVectorizer
    text_vectorized = count_vectorizer.transform([preprocessed_text])
    
    return text_vectorized

def recommend_movies(movie_title):
    # Vectorize the movie title
    movie_vector = vectorize_text(movie_title)
    
    # Compute cosine similarity between the input movie and all movies in the dataset
    cosine_similarities = cosine_similarity(movie_vector, count_vectorizer.transform(filledna['soup']))
    
    # Get movie indices based on similarity scores
    movie_indices = cosine_similarities.argsort()[0][::-1]
    
    # Get top 10 similar movies
    top_movies = [(filledna['Name of movie'].iloc[idx], cosine_similarities[0][idx]) for idx in movie_indices[:10]]
    
    return top_movies

def main():
    # Set page configuration
    set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎥",
        layout="wide"
    )

    # Use CSS to set background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://wallpaperswide.com/download/dark_gothic_lion-wallpaper-1280x800.jpg');  # Replace with your background image URL
            background-size: cover;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Movie Recommendation System")

    # Input field for movie title
    movie_title = st.text_input("Enter a movie title")

    if st.button("Get Recommendations"):
        if movie_title:
            recommendations = recommend_movies(movie_title)
            st.write("Top 10 Recommended Movies:")
            for movie, similarity_score in recommendations:
                st.write(f"- {movie} (Similarity Score: {similarity_score:.2f})")
        else:
            st.write("Please enter a movie title to get recommendations.")

if __name__ == '__main__':
    main()
