import streamlit as st
import numpy as np
import pandas as pd
import emoji as emoji
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import re
import string

# Load the trained model
with open("model.json", "r") as file:
    model_json = file.read()
model = model_from_json(model_json)
model.load_weights("model.weights.h5")

# Load GloVe embeddings
embeddings = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coeffs

# Emoji dictionary
emoji_dictionary = {
    "0": "\u2764\uFE0F",    # :heart:
    "1": ":baseball:",
    "2": ":beaming_face_with_smiling_eyes:",
    "3": ":downcast_face_with_sweat:",
    "4": ":fork_and_knife:",
}

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

# Get output embeddings
def getOutputEmbeddings(X):
    embedding_matrix_output = np.zeros((1, 10, 50))
    X = X.split()
    for jx in range(len(X)):
        embedding_matrix_output[0][jx] = embeddings.get(X[jx].lower(), np.zeros((50,)))
    return embedding_matrix_output

# Streamlit app
st.title("Emoji Prediction App")
st.write("Enter a sentence and get the corresponding emoji!")

# Input from user
user_input = st.text_input("Enter a sentence:")

if st.button("Predict Emoji"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        emb_input = getOutputEmbeddings(preprocessed_text)
        prediction = model.predict(emb_input)
        emoji_code = np.argmax(prediction, axis=1)[0]
        emoji_output = emoji.emojize(emoji_dictionary[str(emoji_code)])
        st.write(f"Predicted Emoji: {emoji_output}")
    else:
        st.write("Please enter a sentence.")
