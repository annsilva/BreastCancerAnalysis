import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

import pickle

# Load the breast cancer dataset
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names, data.target_names

# Make predictions
def predict(model, scaler, features, selector):
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_scaled = selector.transform(features_scaled)
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit app
def main():
    st.title("Breast Cancer Prediction App")
    
    # Load data and train model
    df, feature_names, target_names = load_data()
    X = df.drop('target', axis=1)

    model = pickle.load(open('best_mlp.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    selector = pickle.load(open('selector.pkl', 'rb'))
    
    # User input
    st.sidebar.header("Input Features")
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.sidebar.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    
    # Make prediction
    if st.sidebar.button("Predict"):
        input_features = np.array(list(user_input.values()))
        prediction = predict(model, scaler, input_features, selector)
        st.write(f"Prediction: {target_names[prediction]}")
    
    # Display dataset info
    st.subheader("Dataset Information")
    st.write(df.describe())
    

if __name__ == "__main__":
    main()