import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import streamlit as st
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stopword = stopwords.words('english')


st.set_page_config(page_title="Patient Condition Classifier", page_icon="💊")
st.title('🩺 Patient Condition Classifier 😷💊💉')
st.markdown("Classify a patient's condition from their description and get top-rated drug suggestions.")



vector = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('model.pkl')
data = pd.read_csv('dataset\drugsComTrain_raw.csv')
data.columns = data.columns.str.strip() 


st.subheader('📝 Enter patient condition description:')
input_text = st.text_area('Description',height=200, placeholder="e.g. I have a headache and fever.")



def review_of_words(review):
    review = review.lower()
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    review = review.split()
    review = [word for word in review if word not in stopword]
    review = [lemmatizer.lemmatize(word) for word in review]
    return ' '.join(review)



def predict(text):
    processed_text = review_of_words(text)
    transformed = vector.transform([processed_text])
    return model.predict(transformed)[0]



def top_drug_extractor(condition, data):
    df_top = data[(data['rating'] >= 9) & (data['usefulCount'] >= 100)]
    df_top = df_top[df_top['condition'] == condition]
    df_top = df_top.sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    return df_top['drugName'].drop_duplicates().head(3).tolist()



if st.button('Predict Condition'):
    if input_text.strip():
        condition = predict(input_text)
        st.success(f'🧾 Predicted condition: **{condition}**')

        top_drugs = top_drug_extractor(condition, data)
        if top_drugs:
            st.subheader("💊 Top Recommended Drugs:")
            for i, drug in enumerate(top_drugs, start=1):
                st.markdown(f"{i}. **{drug}**")
        else:
            st.warning("No top drugs found for this condition.")
    else:
        st.error("⚠️ Please enter a valid condition description.")
