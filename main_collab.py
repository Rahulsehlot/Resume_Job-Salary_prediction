
import nltk
# nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


from PyPDF2 import PdfReader
reader = PdfReader('1CR21MC082_RAHULJAIN_RESUME (5).pdf')
page=reader.pages[0]
resume = page.extract_text()
print(resume)

tokens = word_tokenize(resume)
print(tokens)

stop_words = set(stopwords.words('english'))
print(stop_words)
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print(filtered_tokens)

job_scores = {
    'Software engineer': {'Python': 10, 'Java': 10, 'database management': 8, 'web development': 5},
    'Web developer': {'HTML': 8, 'CSS': 8, 'JavaScript': 10, 'web development': 10},
    'Finance analyst': {'Financial analysis': 10, 'Accounting': 8, 'Excel': 7, 'Business intelligence': 6},
    'Marketing specialist': {'Marketing strategy': 10, 'Social media': 8, 'SEO': 7, 'Content creation': 6}
}

suitability_scores = {}
for job, skills in job_scores.items():
    job_score = 0
    for skill, score in skills.items():
        print(skill)
        if skill in filtered_tokens:
            print("=====",skill,"=====")
            job_score += score
            print("=====",score,"=====")
    suitability_scores[job] = job_score
print(job_score)

best_job = max(suitability_scores, key=suitability_scores.get)
print(best_job)
print(suitability_scores)

import pandas as pd
df = pd.DataFrame(pd.DataFrame(suitability_scores.items()))
print(df)
# print(df.info())
# print(df.head())
# print(df.describe())
import matplotlib.pyplot as plt

df.plot(x=0, y=1, kind='scatter')

df.plot(x =0, y=1, kind='line')

df.plot(x =0, y=1, kind='bar')

plt.pie(df[1], labels=df[0])