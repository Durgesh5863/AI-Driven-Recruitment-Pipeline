#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import os
import numpy as np
import textstat
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# Prediction

# In[3]:


pred_data = pd.read_excel("Copy of prediction_data.xlsx")


# Vectorizer model

# In[4]:


with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


# Sentence Transformer Model

# In[5]:


# Load the SentenceTransformer model
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')


# Classifier Model

# In[6]:


# Load the saved model
with open('sentence_transformer_model.pkl', 'rb') as file:
    classifier = pickle.load(file)


# I need 18 features
# 
# 'num_words_in_transcript',
#     'resume_jd_similarity', 'resume_transcript_similarity', 'sentiment',
#     'transcript_length', 'resume_length',
#     'job_description_experience_match', 'text_complexity_transcript',
#     'text_complexity_resume', 'lexical_diversity', 
#     'technical_skill_match', 'soft_skills_sentiment',
#     'cultural_fit_sentiment', 'job_fit_score', 'confidence_score',
#     'clarity_score', 'job_desc_complexity', 'interaction_quality'

# In[7]:


# Calculate resume and job description similarity (Cosine Similarity)
vectorizer = TfidfVectorizer()
resume_jd_similarity = []
for i in range(len(pred_data)):
    resume = pred_data['Resume'][i]
    jd = pred_data['Job Description'][i]
    similarity = cosine_similarity(tfidf_vectorizer.transform([resume, jd]))[0, 1]
    resume_jd_similarity.append(similarity)
pred_data['resume_jd_similarity'] = resume_jd_similarity


# In[8]:


# Calculate resume and transcript similarity (Cosine Similarity)
resume_transcript_similarity = []
for i in range(len(pred_data)):
    resume = pred_data['Resume'][i]
    transcript = pred_data['Transcript'][i]
    similarity = cosine_similarity(tfidf_vectorizer.transform([resume, transcript]))[0, 1]
    resume_transcript_similarity.append(similarity)
pred_data['resume_transcript_similarity'] = resume_transcript_similarity


# In[9]:


# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Lists to store sentiment results
sentiments = []
polarity = []

# Perform sentiment analysis on each transcript
for i in range(len(pred_data)):
    transcript = pred_data['Transcript'][i]
    sentiment_score = sia.polarity_scores(transcript)
    sentiments.append(sentiment_score['compound'])  # Compound sentiment score
    polarity.append('positive' if sentiment_score['compound'] > 0 
                    else 'negative' if sentiment_score['compound'] < 0 
                    else 'neutral')

# Add the results to the DataFrame
pred_data['sentiment'] = sentiments
pred_data['polarity'] = polarity

# Count of each sentiment category
polarity_counts = pred_data['polarity'].value_counts()
print("\n2. Sentiment Polarity Distribution:")
for polarity, count in polarity_counts.items():
    print(f"   - {polarity.capitalize()}: {count} occurrences ({(count / len(pred_data) * 100):.2f}%)")

# Overall average sentiment score
average_sentiment = pred_data['sentiment'].mean()
print(f"\n3. Overall Average Sentiment Score: {average_sentiment:.2f}")
if average_sentiment > 0:
    print("   - The overall sentiment of the transcripts is positive.")
elif average_sentiment < 0:
    print("   - The overall sentiment of the transcripts is negative.")
else:
    print("   - The overall sentiment of the transcripts is neutral.")


# In[10]:


# Function to calculate lexical diversity
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

# Compute lexical diversity for each transcript
pred_data['lexical_diversity'] = pred_data['Transcript'].apply(lexical_diversity)

# Calculate statistics
average_diversity = pred_data['lexical_diversity'].mean()


# In[11]:


# Length of transcript (number of words)
pred_data['transcript_length'] = pred_data['Transcript'].apply(lambda x: len(x.split()))

# Calculate statistics
average_length = pred_data['transcript_length'].mean()
min_length = pred_data['transcript_length'].min()
max_length = pred_data['transcript_length'].max()


# In[12]:


# Function to compute similarity score between Resume and Job Description
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Calculate technical skill matching score
pred_data['technical_skill_match'] = pred_data.apply(lambda row: compute_similarity(row['Resume'], row['Job Description']), axis=1)


# In[13]:


#Soft Skills
pred_data['soft_skills_sentiment'] = pred_data['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[14]:


# Resume length (number of words)
pred_data['resume_length'] = pred_data['Resume'].apply(lambda x: len(x.split()))


# In[15]:


# Job Description Experience Match (Simple matching based on keywords, could be improved)
pred_data['job_description_experience_match'] = pred_data.apply(lambda row: len(set(row['Resume'].split()) & set(row['Job Description'].split())), axis=1)


# In[16]:


#Cultural fit sentiment
pred_data['cultural_fit_sentiment'] = pred_data['Reason for decision'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[17]:


#job score
def job_fit_analysis(job_desc, transcript):
    # You can use similarity or keyword matching here
    job_keywords = job_desc.split()
    transcript_keywords = transcript.split()
    common_keywords = set(job_keywords).intersection(transcript_keywords)
    return len(common_keywords) / len(job_keywords)

pred_data['job_fit_score'] = pred_data.apply(lambda row: job_fit_analysis(row['Job Description'], row['Transcript']), axis=1)


# In[18]:


import re

# Define a function to calculate the confidence score
def calculate_confidence_score(text):
    # Count occurrences of "I think" and "Maybe" (case-insensitive)
    confidence_phrases = re.findall(r'\b(I think|Maybe)\b', text, flags=re.IGNORECASE)
    return len(confidence_phrases)

# Apply the function to calculate confidence scores
pred_data['confidence_score'] = pred_data['Transcript'].apply(calculate_confidence_score)


# In[19]:


#job description complexity
pred_data['job_desc_complexity'] = pred_data['Job Description'].apply(lambda x: textstat.flesch_reading_ease(x))


# In[20]:


#interaction quality
pred_data['interaction_quality'] = pred_data['num_words_in_transcript'] * pred_data['sentiment']


# In[21]:


pred_data['clarity_score'] = pred_data['Transcript'].apply(lambda x: textstat.flesch_reading_ease(x))


# In[22]:


# Text complexity (resume and transcript - using a simple metric like Flesch Reading Ease)
def text_complexity(text):
    # Implement text complexity (e.g., Flesch Reading Ease)
    # Here's a placeholder function:
    return len(text.split()) / len(set(text.split()))  # A basic metric

pred_data['text_complexity_transcript'] = pred_data['Transcript'].apply(text_complexity)
pred_data['text_complexity_resume'] = pred_data['Resume'].apply(text_complexity)


# In[23]:


pred_data.columns


# In[ ]:





# In[24]:


# Define a function to convert the text into embeddings
def get_embeddings(texts, model):
    return np.array([model.encode(text) for text in texts])


# In[25]:


# Get embeddings for 'Transcript', 'Job Description', and 'Resume'
transcript_embeddings = get_embeddings(pred_data['Transcript'], sentence_transformer)
job_desc_embeddings = get_embeddings(pred_data['Job Description'], sentence_transformer)
resume_embeddings = get_embeddings(pred_data['Resume'], sentence_transformer)
reason_embeddings = get_embeddings(pred_data['Reason for decision'], sentence_transformer)
polarity_embeddings = get_embeddings(pred_data['polarity'], sentence_transformer)


# In[26]:


num_features_array = pred_data[
    [
        'num_words_in_transcript', 'resume_jd_similarity', 
        'resume_transcript_similarity', 'sentiment',
        'lexical_diversity', 'transcript_length', 'technical_skill_match',
        'soft_skills_sentiment', 'resume_length',
        'job_description_experience_match', 'cultural_fit_sentiment',
        'job_fit_score', 'confidence_score', 'job_desc_complexity',
        'interaction_quality', 'clarity_score', 
        'text_complexity_transcript', 'text_complexity_resume'
    ]
].to_numpy()


# Concatenate the embeddings
features = np.concatenate([transcript_embeddings, resume_embeddings, reason_embeddings, job_desc_embeddings, polarity_embeddings, num_features_array], axis=1)


# In[27]:


# Predict decision using embedding features
embed_decision = classifier.predict(features)


# In[28]:


print("Predicted decision:", embed_decision)


# In[29]:


# Add predictions to the DataFrame
pred_data['Decision'] = ['Accept' if pred == 1 else 'Reject' for pred in embed_decision]


# In[30]:


pred_data.to_csv('prediction.csv')


# In[ ]:





# Send Mail

# In[31]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# In[32]:


# Define a function to send an email
def send_email_with_attachment(sender_email, sender_password, to_email, subject, body, file_path):
    try:
        # Create the email message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        # Attach the file
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(file_path)}",
            )
            message.attach(part)
        else:
            print(f"File not found: {file_path}")

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Start TLS encryption
            server.login(sender_email, sender_password)  # Log in to the email server
            server.send_message(message)  # Send the email
            print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")

# Email details
sender_email = "durgeshbabu5863@gmail.com"
sender_password = "bdtc cjfu vvro afdx"  # App-specific password
to_email = "malathula00000@gmail.com"
subject = "Predicted Outcome of the Candidates"
body = "Here is the attached pdf which consists of the predicted outcome of the candidates"
file_path = r"D:\Internship - Infosys\Project\prediction.csv"

# Call the function to send the email
send_email_with_attachment(sender_email, sender_password, to_email, subject, body, file_path)


# In[ ]:




