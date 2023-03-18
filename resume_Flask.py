from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump, load

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    file = request.files['file']
    algorithm = request.form['algorithm']
    if file and file.filename.endswith('.pdf'):
        # Convert resume PDF to text
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file)
        resume_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            resume_text += page.extract_text()

        # Predict job and salary
        if algorithm == 'Decision Tree':
            predicted_job, predicted_salary = predict_job_and_salary(resume_text, job_classifier)
            print(predicted_job, predicted_salary)
        elif algorithm == 'Random Forest':
            predicted_job, predicted_salary = predict_job_and_salary_rf(resume_text)
            print(predicted_job, predicted_salary)
        # Display predictions
        return render_template('result.html', job_title=predicted_job, salary='${:,.2f}'.format(predicted_salary))
    else:
        return 'Please upload a resume in PDF format'

if __name__ == '__main__':
    app.run(debug=True)
