import streamlit as st # type: ignore
import pandas as pd
import pickle

st.set_page_config(page_title="Career Role Predictor", layout="centered")
st.title("💼 AI Career Role Predictor")
st.write("Enter your professional details below to discover your optimal career path.")

@st.cache_resource
def load_artifacts():
    with open('career_predictor_model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    artifacts = load_artifacts()
    vectorizer = artifacts['vectorizer']
    model = artifacts['model']
except FileNotFoundError:
    st.error("Artifacts file 'career_predictor_model.pkl' not found. Please run your training script first!")
    st.stop()

st.subheader("Your Profile Details")

skills_input = st.text_input(
    "Skills (comma-separated)", 
    placeholder="e.g., Python, SQL, TensorFlow, Machine Learning"
)
qualification_input = st.text_input(
    "Highest Qualification", 
    placeholder="e.g., Master's in Data Science"
)
experience_input = st.selectbox(
    "Experience Level", 
    options=["Entry", "Mid", "Senior"],
    index=0
)

if st.button("Predict Career Path", type="primary"):
    if not skills_input or not qualification_input:
        st.warning("Please provide both skills and qualifications to get an accurate prediction.")
    else:
        combined_text = f"{skills_input} {qualification_input} {experience_input}"
        
        transformed_input = vectorizer.transform([combined_text])
        
        prediction = model.predict(transformed_input)[0]
        
        probabilities = model.predict_proba(transformed_input)[0]
        classes = model.classes_
        
        prob_df = pd.DataFrame({
            'Career Role': classes,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=False)
        
        top_5_df = prob_df.head(5).reset_index(drop=True)
        
        st.success(f"### 🎯 Recommended Career Match: **{prediction}**")
        
        st.write("---")
        st.subheader("Top 5 Career Role Probabilities")
        
        st.dataframe(
            top_5_df.style.format({'Probability': '{:.2%}'}),
            use_container_width=True
        )