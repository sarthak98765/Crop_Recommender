import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('Notebook/data/Crop_recommendation.csv')
df.rename(columns={'label': 'Crop'}, inplace=True)

# Selection of Feature and Target variables.

x = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['Crop']
y = pd.get_dummies(target)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier

# Training

knn_clf = KNeighborsClassifier()
model = MultiOutputClassifier(knn_clf, n_jobs=-1)
model.fit(x_train, y_train)

columns = x.columns.tolist()

st.set_page_config(page_title="Farmer assister")
st.title('Crop Recommendor')

N = st.text_input('Nitrogen Content in soil')
P = st.text_input('Phosphorus content in soil')
K = st.text_input('Potassium content in soil')
temp = st.text_input('Temprature')
humidity = st.text_input('Humidity')
pH = st.text_input('pH')
rainfall = st.text_input('Rainfall')

def predict():
    a = float(N)
    b = float(P)
    c = float(K)
    d = float(temp)
    e = float(humidity)
    f = float(pH)
    g = float(rainfall)

    features = np.array([[a, b, c, d, e, f, g]])

    crop_recommendation = model.predict(features)

    # Get the column index with value 1
    column_index = np.argmax(crop_recommendation)

    # List of column names (replace with your actual column names)
    column_names = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                    'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    column_names.sort()

    # Get the column name with value 1
    recommended_crop = column_names[column_index]

    st.write("Recommended crop: ", recommended_crop)

if st.button('Predict'):
    predict()
