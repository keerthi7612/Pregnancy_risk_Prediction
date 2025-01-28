import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load the dataset
def load_data(file_path):
    data = pd.read_csv("D:/Working-project/finalData.csv")
    return data


# Load the prediction model
def load_model(file_path):
    with open("D:/Working-project/trained_model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model


# Make predictions
def make_prediction(model, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict([input_array])
    return prediction


# Main function for the Streamlit app
def main():
    st.title("Final Stage Web Application with Prediction")


    # Sidebar for navigation
    menu = ["Home", "Guidance", "Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.header("Welcome")
        st.write("This is the home page for the web application.")

    elif choice == "Guidance":
        st.header("Guidance")

        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV file for analysis:", type="csv")
        
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            st.subheader("Dataset Overview")
            st.dataframe(data)

            st.subheader("Data Summary")
            st.write(data.describe())

            st.subheader("Shape of the Data")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

            # Column Selection for Visualization
            st.subheader("Column-wise Visualization")
            column_to_visualize = st.selectbox("Select a column to visualize:", data.columns)
            if column_to_visualize:
                st.line_chart(data[column_to_visualize])
        
    
    elif choice == "Prediction":
        st.header("Prediction")

        model_path = "trained_model.pkl"
        model=load_model(model_path)

        # File uploader for the model
        #model_file = st.file_uploader("Upload your trained model file (Pickle format):", type="pkl")
        #model = load_model("trained_model.pkl")
        
        #if model is not None:
           # model = load_model(model_file)

        st.subheader("Enter Input Data for Prediction")
        st.write("Please provide the following details:")

        age = st.number_input("Age", min_value=0, step=1)
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0.0, step=0.1)
        diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, step=0.1)
        bs = st.number_input("Blood Sugar Level", min_value=0.0, step=0.1)
        body_temp = st.number_input("Body Temperature (°C)", min_value=0.0, step=0.1)
        heart_rate = st.number_input("Heart Rate", min_value=0, step=1)

        input_data = [age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]

        if st.button("Predict"):
                prediction = make_prediction(model, input_data)
                # Interpret prediction
                risk_mapping = {0: "Low Risk", 1: "High Risk"}
                readable_prediction = risk_mapping.get(prediction[0], "Unknown")
                st.success(f"The predicted Risk Level is: {readable_prediction}")

if __name__ == "__main__":
    main()