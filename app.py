import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import category_encoders as ce

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("TunisianUsedCars_Cleaned.csv")
    df = remove_outliers(df)
    return df

# Function to remove outliers
def remove_outliers(df):
    def find_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)

    mask = pd.Series([True] * len(df))
    for column in ['Price', 'Mileage', 'Puissance Fiscale']:
        mask = mask & ~find_outliers_iqr(df, column)
    return df[mask]

# Preprocessing and model training
@st.cache_data
def preprocess_and_train(df):
    categorical_columns = ["model"]
    numerical_columns = ["Mileage", "year", "Puissance Fiscale"]

    # Target encoding for categorical columns
    encoder = ce.TargetEncoder(cols=categorical_columns)
    df_encoded = encoder.fit_transform(df[categorical_columns], df['Price'])
    df_final = pd.concat([df_encoded, df[numerical_columns]], axis=1)

    # Define features and target
    X = df_final
    y = df['Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=5)
    rf_model.fit(X_train, y_train)

    return rf_model, encoder

# Prediction function
def predict_price(model, encoder, inputs):
    # Convert inputs to a DataFrame
    inputs_df = pd.DataFrame([inputs], columns=["model", "Mileage", "year", "Puissance Fiscale"])
    
    # Separate the categorical column for encoding
    categorical_input = inputs_df[["model"]]
    encoded_categorical = encoder.transform(categorical_input)
    
    # Combine encoded categorical features with numerical columns
    final_features = pd.concat([encoded_categorical, inputs_df[["Mileage", "year", "Puissance Fiscale"]]], axis=1)
    
    # Make the prediction
    price = model.predict(final_features)
    return price[0]

# Load data and train model
st.title("Tunisian Car Price Prediction")
df = load_data()
model, encoder = preprocess_and_train(df)

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    body {
        background-image: url('file://DSC00533e-1024x680.jpg');
        background-size: cover;
        color: #fff;
    }
    .css-18e3th9 {
        background-color: rgba(0, 0, 0, 0.5); /* Adds transparency to the sidebar */
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs for prediction
st.sidebar.header("Enter Car Features for Prediction")
mileage = st.sidebar.number_input("Mileage (km)", min_value=0, step=1000)
year = st.sidebar.number_input("Year", min_value=1990, max_value=2024, step=1)
puissance_fiscale = st.sidebar.number_input("Puissance Fiscale", min_value=0, step=1)
model_input = st.sidebar.selectbox("Car Model", df["model"].unique())

# Make prediction
if st.sidebar.button("Predict Price"):
    inputs = [model_input, mileage, year, puissance_fiscale]
    price = predict_price(model, encoder, inputs)
    st.write(f"### Predicted Price: {price:.2f} TND")
