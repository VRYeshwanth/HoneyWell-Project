import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('skills_dataset.csv')

df['combined_features'] = df['skills'] + " " + df['qualification'] + " " + df['experience_level']

X = df['combined_features']
y = df['job_role']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Vectorizing features using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training the MultinomialNB model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Calculate accuracy on test data
y_pred = nb_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")

artifacts = {
    'vectorizer': vectorizer,
    'model': nb_model,
    'accuracy': accuracy
}

model_filename = 'career_predictor_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(artifacts, file)

print(f"Success! Model and vectorizer saved as '{model_filename}'")
