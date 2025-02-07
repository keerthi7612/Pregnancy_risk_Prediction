import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
model=LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


classifier = SVC(class_weight='balanced')


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
    prediction = model.predict(input_array)
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

        model_path = "D:/Working-project/trained_model.pkl"
        model=load_model(model_path)


        data = load_data("D:/Working-project/finalData.csv")
        target_column = "Risk Level"
        x = data.drop(columns=[target_column])


        scaler_path = "D:/Working-project/scaler.pkl"  
        with open(scaler_path, 'rb') as file:
         scaler = pickle.load(file)


        encoder_path = "D:/Working-project/label_encoder.pkl"
        with open(encoder_path, 'rb') as file:
         encoder = pickle.load(file)

        #std_data = scaler.transform(input_data) 
        # File uploader for the model
        #model_file = st.file_uploader("Upload your trained model file (Pickle format):", type="pkl")
        #model = load_model("trained_model.pkl")
        
        #if model is not None:
           # model = load_model(model_file)

        st.subheader("Enter Input Data for Prediction")
        st.write("Please provide the following details:")

        age = st.number_input("Age",min_value=0, step=1, format="%d")
        systolic_bp = st.number_input("Systolic Blood Pressure",min_value=0, step=1, format="%d")
        diastolic_bp = st.number_input("Diastolic Blood Pressure",min_value=0, step=1, format="%d")
        bs = st.number_input("Blood Sugar Level",min_value=0, step=1, format="%d")
        body_temp = st.number_input("Body Temperature (°C)",min_value=0, step=1, format="%d")
        heart_rate = st.number_input("Heart Rate",min_value=0, step=1, format="%d")

        feature_names =['Age', 'Systolic BP', 'Diastolic BP', 'BS', 'Body Temp', 'Heart Rate']

        #x_train = pd.DataFrame(x_train, columns=feature_names)
        #scaler = StandardScaler()
        #x_train = scaler.fit_transform(x_train)
        #pickle.dump(scaler, open("D:/Working-project/scaler.pkl", "wb"))


        input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]], columns=feature_names)


        if st.button("Predict"):

                std_data = scaler.transform(input_data)

                prediction =model.predict(std_data)

                #prediction = make_prediction(model, input_data)
                # Interpret prediction
                #risk_mapping = {0: "Low Risk", 1: "High Risk"}
                #readable_prediction = risk_mapping.get(prediction[0], "Unknown")
                #st.success(f"The predicted Risk Level is: {readable_prediction}")

                predicted_label = encoder.inverse_transform(prediction)


                st.success(f"🔹 The predicted **Risk Level** is: **{predicted_label[0]}**")





if __name__ == "__main__":
    main()