import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('skills_dataset.csv')

df['combined_features'] = df['skills'] + " " + df['qualification'] + " " + df['experience_level']

X = df['combined_features']
y = df['job_role']

print("Vectorizing features using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

print("Training the MultinomialNB model...")
nb_model = MultinomialNB()
nb_model.fit(X_tfidf, y)

artifacts = {
    'vectorizer': vectorizer,
    'model': nb_model
}

model_filename = 'career_predictor_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(artifacts, file)

print(f"Success! Model and vectorizer saved as '{model_filename}'")