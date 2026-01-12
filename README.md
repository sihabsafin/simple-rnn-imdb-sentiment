## 🎬 AI Movie Sentiment Analyzer

End-to-End NLP Project using Simple RNN & Word Embeddings

## 📌 Project Overview

This project is an end-to-end Natural Language Processing (NLP) application that analyzes the sentiment of movie reviews using a Simple Recurrent Neural Network (RNN) with word embeddings, and deploys the trained model as a production-ready Streamlit web application.

The goal of this project was not only to train a neural network, but to take a deep learning model all the way to deployment, handling real-world challenges such as model compatibility, preprocessing consistency, and user-focused design.

👉 Live App: https://rnn-movie-sentiment-analyzer.streamlit.app/

## 🎯 Problem Statement

Movie reviews are unstructured text data.
The task is to determine whether a given review expresses a positive, negative, or uncertain sentiment.

Challenges addressed in this project:

Converting raw text into numerical representations

Capturing sequential information using RNNs

Handling real-world deployment issues with legacy models

Presenting model predictions in a clear, interpretable, and user-friendly way

## 🧠 Model & Approach
🔹 Dataset

IMDB Movie Reviews Dataset

Binary sentiment labels (positive / negative)

Vocabulary size limited for efficient embedding learning

## 🔹 Model Architecture

Embedding Layer – converts word indices into dense vector representations

Simple RNN Layer – captures sequential dependencies in text

Dense Output Layer (Sigmoid) – outputs sentiment probability

This model was intentionally kept simple and interpretable to focus on understanding the full NLP pipeline rather than chasing benchmark scores.

## 🔄 Text Processing Pipeline

The application follows the same preprocessing steps used during training:

Convert input text to lowercase

Tokenize words using the IMDB word index

Map words to integer indices

Pad sequences to a fixed length

Feed the processed sequence into the RNN model

A dedicated “See how text is processed” section in the UI explains this pipeline for transparency.

## 🌐 Web Application Features

This project goes beyond a basic demo and focuses on real product-style features:

##  Core Features

Clean text input with live character count

One-click example reviews (positive, negative, ambiguous)

Sentiment prediction with confidence percentage

Sentiment banding:

🟢 Strong Positive

🟡 Neutral / Uncertain

🔴 Negative

🎨 UI & UX

Dark / Light mode toggle

Movie-themed modern interface

Animated confidence progress bar

Responsive layout (desktop-friendly)

## 🔍 Interpretability

Lightweight sentence-level insights (rule-based keyword detection)

Explicit model limitation notice:

Simple RNN may struggle with negation and long sentences

## 🧾 Practical Features

Download prediction result as CSV

Session-based prediction history

Timestamped results

## ⚠️ Model Limitations (Important)

This project intentionally highlights limitations instead of hiding them:

Simple RNNs struggle with long-range dependencies

Negation handling is limited (e.g., “not good”)

No attention mechanism or transformer architecture is used

These limitations are explicitly communicated in the UI, reflecting a professional and honest ML mindset.

## 🛠️ Tech Stack

Python

TensorFlow / Keras

Streamlit

NumPy & Pandas

IMDB Dataset

##  Deployment Notes

During deployment, the original legacy .h5 model format caused compatibility issues with modern Keras runtimes.
To solve this, the model architecture was reconstructed and migrated into the modern .keras format, ensuring:

Forward compatibility with Keras 3

Stable deployment on Streamlit Cloud

No retraining required

This reflects a real-world ML lifecycle challenge and its practical resolution.

## 📁 Repository Structure
```bash
simple-rnn-imdb-sentiment/
│
├── app.py                   # Streamlit application
├── simple_rnn_imdb.keras    # Trained model (modern format)
├── requirements.txt
├── runtime.txt
├── README.md
└── LICENSE
