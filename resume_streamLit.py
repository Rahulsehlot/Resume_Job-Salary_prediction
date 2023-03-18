import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump, load
from PyPDF2 import PdfFileReader

# Load dataset
data = pd.read_csv('jobss.csv')

# Data preprocessing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Key Skills'])
y_job = data['Job Title']
y_salary = data['sal']

# Train Decision Tree Classifier for job prediction
data['Job Title'].fillna(data['Job Title'].mode()[0], inplace=True)
job_classifier = DecisionTreeClassifier()
job_classifier.fit(X, y_job)

# Train Random Forest Classifier for job prediction
rf_job_classifier = RandomForestClassifier(n_estimators=42)
rf_job_classifier.fit(X, y_job)

# Train Decision Tree Regressor for salary prediction
salary_regressor = DecisionTreeRegressor()
salary_regressor.fit(X, y_salary)

# Train Random Forest Regressor for salary prediction
rf_salary_regressor = RandomForestRegressor(n_estimators=55)
rf_salary_regressor.fit(X, y_salary)

# Save trained models
dump(job_classifier, 'job_classifier.joblib')
dump(rf_job_classifier, 'rf_job_classifier.joblib')
dump(salary_regressor, 'salary_regressor.joblib')
dump(rf_salary_regressor, 'rf_salary_regressor.joblib')

# Load models
job_classifier = load('job_classifier.joblib')
rf_job_classifier = load('rf_job_classifier.joblib')
salary_regressor = load('salary_regressor.joblib')
rf_salary_regressor = load('rf_salary_regressor.joblib')

# Define function to predict job and salary
def predict_job_and_salary(resume_text, classifier):
    new_resume_features = vectorizer.transform([resume_text])
    predicted_job = classifier.predict(new_resume_features)[0]
    predicted_salary = salary_regressor.predict(new_resume_features)[0]
    return predicted_job, predicted_salary

# Define function to predict job and salary using random forest classifier
def predict_job_and_salary_rf(resume_text):
    predicted_job = rf_job_classifier.predict(vectorizer.transform([resume_text]))[0]
    predicted_salary = rf_salary_regressor.predict(vectorizer.transform([resume_text]))[0]
    return predicted_job, predicted_salary

# Streamlit app
st.set_page_config(page_title='Resume Analyzer')
st.title('Resume Analyzer')

# Get resume file from user
file = st.file_uploader('Upload resume in PDF format', type=['pdf'])

# Get prediction algorithm from user
algorithm = st.selectbox('Select prediction algorithm', ('Decision Tree', 'Random Forest'))

if st.button('Predict') and file is not None:
    from PyPDF2 import PdfReader
    # Convert resume PDF to text
    pdf_reader = PdfReader(file)
    resume_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        resume_text += page.extract_text()



    # Predict job and salary
    if algorithm == 'Decision Tree':
        predicted_job, predicted_salary = predict_job_and_salary(resume_text, job_classifier)
    elif algorithm == 'Random Forest':
        predicted_job, predicted_salary = predict_job_and_salary_rf(resume_text)

    # Display predictions
    st.write('Job Title Prediction:', predicted_job)
    st.write('Salary Prediction: ${:,.2f}'.format(predicted_salary))
else:
    st.write('Please upload a resume in PDF format')
