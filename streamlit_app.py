import streamlit as st
import pickle
# from sklearn.svm import SVC
# import pandas as pd
# import numpy as np
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer  = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Reviews App")
st.write('This app will tell you if a customer review is positive, neutral or negative!')

st.header("Type a review!")
review = st.text_input("type your review here", "e.g.: This product is amazing!")
if st.button("Analyze review"):
    st.write("Analyzing review...")
    def data_cleaning(text):
        """
        This function processes each setence and applies regex patterns to remove undesired characters.
        In this case we built it detele characters that should be equally translated by computers and humans:
        - special characters
        - numerical characters/digits
        - single characthers
        - multiple spaces (for cleaning purposes)

        Argument: text/corpus/document/sentence; string
        """

        # Remove numbers
        text_no_special_characters = re.sub(r'[^A-Za-z\s]+', ' ', str(text))

        # Remove all single characters (e.g., 'a', 'b', 'c' that appear as standalone)
        text_no_single_charac = re.sub(r'\b\w\b', '', text_no_special_characters)

        # Clean up extra spaces left after removing single characters
        text_cleaned = re.sub(r'\s+', ' ', text_no_single_charac).strip()

        # Transform data to lowercase
        text_cleaned = text_cleaned.lower()

        return text_cleaned


    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""

        tag = nltk.pos_tag([word])[0][1][0]
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)

    def data_processing(text):
        """
        This function processes each sentence in the following order:
        1. Tokenize each word of the sentence.
        2. Remove stopwords and stem words, if any word is in the 'stopwords.words("english")' list.
        3. Lemmatize every word not in the stopwords list
        4. Join all the tokens per row, to rebuild the sentences.

        Argument: text/corpus/document/sentence; string
        """
        tolkenize_words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in tolkenize_words if word not in stopwords.words("english")]
        text_processed = ' '.join(lemmatized_words)  # Join the words back into a single string

        return text_processed

    review_cleaned = data_cleaning(review)
    review_processed = data_cleaning(review_cleaned)

    # Load our model from pickle document
    with open('SVC.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('TF-IDF_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    review_tfidf = vectorizer.transform([review_cleaned])
    
    prediction = model.predict(review_tfidf)

    st.write(prediction)
    