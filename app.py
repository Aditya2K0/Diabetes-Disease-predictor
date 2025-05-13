import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

csv_path = r'C:\Users\ASUS\Downloads\archive\diabetes.csv'

# Check if the file exists and is not empty
if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    st.error(f"CSV file '{csv_path}' is missing or empty. Please ensure it's present and contains valid data.")
    st.stop()

# Try reading the file
try:
    diabetes_df = pd.read_csv(csv_path)
except pd.errors.EmptyDataError:
    st.error("CSV file is empty. Please check the contents of the file.")
    st.stop()
except Exception as e:
    st.error(f"Failed to read CSV file: {e}")
    st.stop()

# Proceed if data is loaded correctly
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# Streamlit app
def app():
    
    st.title('Diabetes Prediction')

    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    reshaped_input_data = np.asarray(input_data).reshape(1, -1)
    scaled_input = scaler.transform(reshaped_input_data)
    prediction = model.predict(scaled_input)

    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)

    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    app()
