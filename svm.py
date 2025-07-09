from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://akash:akash@cluster0.44hv4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['iris']
collection=db['target_names_pred']


# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Models and Scalers ---
@st.cache_resource
def load_resources():
    """Loads all models and scalers once and caches them."""
    try:
        models = {
            'Binary SVM': joblib.load('svm_binary.pkl'),
            'Multi-class SVM': joblib.load('svm_multi.pkl'),
            'Binary Logistic Regression': joblib.load('logistics_binary.pkl'),
            'Multi-class Logistic Regression (OVR)': joblib.load('logistics_ovr.pkl'),
            'Multi-class Logistic Regression (Multinomial)': joblib.load('logistics_multinomial.pkl')
        }
        scalers = {
            'binary': joblib.load('scaler.pkl'),
            'multi': joblib.load('scaler.pkl')
        }
        iris = load_iris()
        return models, scalers, iris.target_names
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please run the `train_models.py` script first.")
        st.stop()

models, scalers, target_names = load_resources()

# --- Sidebar for User Input ---
st.sidebar.title("Iris Feature Input")
st.sidebar.header("Adjust the sliders to match the flower's measurements.")

def user_input_features():
    """Creates sliders in the sidebar for user input."""
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.2, 0.1)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.3, 0.1)
    
    # Create a DataFrame from the input
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Main Panel for Model Selection and Prediction ---
st.title("🌸 Iris Species Prediction App")
st.write("This app uses five different machine learning models to predict the species of an Iris flower based on its measurements.")

# Display user input
st.subheader("Your Input Features")
st.table(input_df)

# Model selection dropdown
st.subheader("1. Select a Model")
model_choice = st.selectbox(
    "Choose a prediction model:",
    list(models.keys())
)

# Determine if the model is binary or multi-class
is_binary = 'Binary' in model_choice
scaler = scalers['binary'] if is_binary else scalers['multi']
model = models[model_choice]
class_names = target_names[:2] if is_binary else target_names

if is_binary:
    st.info("This is a **binary model** and will only predict between 'setosa' and 'versicolor'.")
else:
    st.info("This is a **multi-class model** and can predict 'setosa', 'versicolor', or 'virginica'.")


# --- Prediction Logic ---
if st.button('Predict Species', type="primary"):
    # Scale the user input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    # Display the result
    st.subheader("2. Prediction Result")
    predicted_class_name = class_names[prediction[0]].capitalize()
    
    st.markdown(f"### The model predicts the species is: **:violet[{predicted_class_name}]**")
    
    # Display prediction probabilities in a more visual way
    st.subheader("Model Confidence")
    st.write("The model assigned the following probabilities to each class:")
    
    # Create a DataFrame for the probabilities
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=[name.capitalize() for name in class_names],
        index=["Probability"]
    ).T
    
    try:
        record_to_save = {
            "timestamp": datetime.now(timezone.utc),
            "inputs": input_df.to_dict('records')[0],
            "model_used": model_choice,
            "predicted_class": predicted_class_name,
            "probabilities": {name: prob for name, prob in zip(class_names, prediction_proba[0])}
        }
        
        collection.insert_one(record_to_save)
        st.success("Prediction record successfully saved to MongoDB!", icon="💾")
        
    except Exception as e:
        st.error(f"Could not save to database: {e}", icon="❌")
    
    # Display as a bar chart
    st.bar_chart(proba_df)

    # Display as a table
    st.table(proba_df.style.format("{:.2%}"))


