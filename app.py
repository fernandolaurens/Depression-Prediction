import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("depression_data.csv")

# Preprocess the data
cat_col = ['Marital Status', 'Education Level', 'Employment Status', 'Smoking Status', 'Family History of Depression']
le = LabelEncoder()
df[cat_col] = df[cat_col].apply(le.fit_transform)

# Scale numeric features
scaler = RobustScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

# Function to train the model and predict the outcome
def train_and_predict(outcome_data):
    # Prepare features and target variable
    X = df[['Age', 'Marital Status', 'Education Level', 'Employment Status', 'Smoking Status', 'Income']]
    y = df['Family History of Depression']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=14)
    knn.fit(X_train, y_train)

    # Make a prediction based on user input (ensure outcome_data is 2D array)
    prediction = knn.predict(np.array(outcome_data).reshape(1, -1))
    
    # Calculate accuracy of the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return prediction, accuracy

# Streamlit interface
st.title("Depression Prediction Dashboard")
st.subheader("Predict the likelihood of depression based on input features")

# User input fields with additional help texts
age = st.number_input("Enter Age", min_value=0, max_value=100, value=30, help="Enter the age of the individual (0-100).")
income = st.number_input("Enter Income", min_value=0, value=30000, help="Enter the annual income of the individual in USD.")

# Define mapping for the categorical variables
education_mapping = {
    0: "High School",
    1: "Bachelor's Degree",
    2: "Master's Degree",
    3: "PhD",
    4: "Other"
}

marital_status_mapping = {
    0: "Single",
    1: "Married",
    2: "Divorced",
    3: "Widowed"
}

employment_mapping = {
    0: "Unemployed",
    1: "Employed"
}

smoking_mapping = {
    0: "Non-smoker",
    1: "Smoker"
}

# Replace dropdown values with descriptive labels in Streamlit form

# Education Level selection with mapped labels
education_level = st.selectbox("Select Education Level", options=list(education_mapping.values()), help="Select the highest level of education achieved.")

# Marital Status selection with mapped labels
marital_status = st.selectbox("Select Marital Status", options=list(marital_status_mapping.values()), help="Select the current marital status.")

# Employment Status selection with mapped labels
employment_status = st.selectbox("Select Employment Status", options=list(employment_mapping.values()), help="Select the current employment status.")

# Smoking Status selection with mapped labels
smoking_status = st.selectbox("Select Smoking Status", options=list(smoking_mapping.values()), help="Select the smoking habit (smoker or non-smoker).")

# Map back to numerical values for prediction
selected_education_level = list(education_mapping.keys())[list(education_mapping.values()).index(education_level)]
selected_marital_status = list(marital_status_mapping.keys())[list(marital_status_mapping.values()).index(marital_status)]
selected_employment_status = list(employment_mapping.keys())[list(employment_mapping.values()).index(employment_status)]
selected_smoking_status = list(smoking_mapping.keys())[list(smoking_mapping.values()).index(smoking_status)]

# Button to trigger prediction
if st.button("Predict"):
    # Prepare the input data for prediction with mapped numerical values
    input_data = [age, selected_marital_status, selected_education_level, selected_employment_status, selected_smoking_status, income]
    
    # Predict and get accuracy
    prediction, accuracy = train_and_predict(input_data)
    
    # Interpret the prediction result
    prediction_label = 'depression' if prediction[0] == 1 else 'not depression'
    
    # Display prediction result and accuracy
    st.success(f"Based on the data entered by the user, it can be concluded that the individual has a history of depression: **{prediction_label}**")
    st.write(f"Model accuracy: **{accuracy:.2f}**")

# Optional: Additional analyses and visualizations

# Show classification report
st.subheader("Model Performance")
if st.checkbox("Show Classification Report"):
    # Prepare features and target variable
    X = df[['Age', 'Marital Status', 'Education Level', 'Employment Status', 'Smoking Status', 'Income']]
    y = df['Family History of Depression']
    
    # Split the data for reporting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=14)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Display the classification report
    st.text("Classification Report:")
    st.text(report)

# Show Age Distribution Plot
if st.checkbox("Show Age Distribution Plot"):
    st.subheader("Distribution of Age in Dataset")
    plt.figure(figsize=(10, 6))
    plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    st.pyplot(plt)
