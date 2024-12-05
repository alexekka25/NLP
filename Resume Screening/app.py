import streamlit as st
import joblib
import re
from PyPDF2 import PdfReader

model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract email address
def extract_email(text):
    email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return email.group(0) if email else "Email not found"

# Function to extract LinkedIn link
def extract_linkedin(text):
    # Match full LinkedIn URLs 
    linkedin = re.search(r'https?://(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?', text)
    if linkedin:
        return linkedin.group(0)
    
    # If no URL is found, looking for "LinkedIn" mentions
    lines = text.split('\n')
    for line in lines:
        if "LinkedIn" in line:
            return line.strip()  
    
    return "LinkedIn link or mention not found"


# Predict category
def predict_category(resume_text):
    resume_vector = vectorizer.transform([resume_text])
    prediction = model.predict(resume_vector)
    category = label_encoder.inverse_transform(prediction)[0]
    return category


def read_resume_file(file):
   
    if file.name.endswith('.pdf'):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    elif file.name.endswith('.txt'):
        return file.read().decode('utf-8', errors='ignore')  
    else:
        st.error("Unsupported file format. Please upload a PDF or TXT file.")
        return None


st.title("Resume Screening App")

uploaded_file = st.file_uploader("Upload your Resume (Text or PDF)", type=["txt", "pdf"])

if uploaded_file:
    resume_text = read_resume_file(uploaded_file)
    
    if resume_text:
        
        email = extract_email(resume_text)
        linkedin = extract_linkedin(resume_text)
        
        
        category = predict_category(resume_text)
        
      
        st.write("### Extracted Information")
        st.write(f"**Email:** {email}")
        st.write(f"**LinkedIn Link:** {linkedin}")
        st.write(f"**Category:** {category}")
