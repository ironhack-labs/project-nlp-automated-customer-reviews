import streamlit as st
import pickle
from sklearn.svm import SVC


  

st.title("Reviews App")
st.write('This app will tell you if a customer review is positive, neutral or negative!')

st.header("Type a review!")
review = st.text_input("type your review here", "e.g.: This product is amazing!")
if st.button("Analyze review"):
    st.write("Analyzing review...")
    # Load our model from pickle document
    with open('SVC.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(review)

    st.write(prediction)
    