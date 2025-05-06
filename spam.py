import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)



v = CountVectorizer()
X = v.fit_transform(df['Message'])
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()

model.fit(X_train, y_train)


st.title("Spam Email Classifier")
st.write("This app classifies emails as spam or not spam.")

input_email = st.text_area("Enter an email message:")

if st.button("Predict"):
    if input_email:
        input_data = v.transform([input_email])
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            
            st.info("This email is classified as *Spam*.")
        else:
            st.balloons()
            st.success("This email is classified as *Not Spam*.")
    else:
        st.write("Please enter an email message to classify.")

hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
