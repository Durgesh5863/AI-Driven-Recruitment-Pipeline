#!/usr/bin/env python
# coding: utf-8

# In[238]:


from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from textblob import TextBlob
import textstat


# In[134]:


data = pd.read_csv('new_combined_data.csv')


# In[3]:


model = SentenceTransformer('all-MiniLM-L6-v2')


# In[4]:


def get_sentence_embeddings_batch(texts, model, batch_size=32):
    """
    Generate sentence embeddings in batches using SentenceTransformer.

    Args:
        texts (list): List of input texts.
        model (SentenceTransformer): Preloaded SentenceTransformer model.
        batch_size (int): Batch size for processing texts.

    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Total number of batches
    print(f"Total Batches: {total_batches}")
    
    for i in range(total_batches):
        # Print progress
        print(f"Processing batch {i + 1}/{total_batches}...")
        
        # Get the current batch
        batch = texts[i * batch_size:(i + 1) * batch_size]
        
        # Generate embeddings for the batch
        batch_embeddings = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


# In[ ]:





# In[5]:


# Apply batch processing
texts = data['Transcript'].tolist()

# Generate BERT embeddings
batch_embeddings = get_sentence_embeddings_batch(texts, model, batch_size=32)

# Convert embeddings to a list of Python lists
batch_embeddings = [emb.tolist() for emb in batch_embeddings]

# Add embeddings as a new column in the DataFrame
data['bert_embeddings_trans'] = batch_embeddings


# In[ ]:





# In[6]:


# Apply batch processing
texts = data['Resume'].tolist()

# Generate BERT embeddings
batch_embeddings = get_sentence_embeddings_batch(texts, model, batch_size=32)

# Convert embeddings to a list of Python lists
batch_embeddings = [emb.tolist() for emb in batch_embeddings]

# Add embeddings as a new column in the DataFrame
data['bert_embeddings_resume'] = batch_embeddings


# In[ ]:





# In[7]:


# Apply batch processing
texts = data['Job Description'].tolist()

# Generate BERT embeddings
batch_embeddings = get_sentence_embeddings_batch(texts, model, batch_size=32)

# Convert embeddings to a list of Python lists
batch_embeddings = [emb.tolist() for emb in batch_embeddings]

# Add embeddings as a new column in the DataFrame
data['bert_embeddings_jd'] = batch_embeddings


# In[ ]:





# In[8]:


# Apply batch processing
texts = data['Reason for decision'].tolist()

# Generate BERT embeddings
batch_embeddings = get_sentence_embeddings_batch(texts, model, batch_size=32)

# Convert embeddings to a list of Python lists
batch_embeddings = [emb.tolist() for emb in batch_embeddings]

# Add embeddings as a new column in the DataFrame
data['bert_embeddings_reason'] = batch_embeddings


# In[ ]:





# In[9]:


# Apply batch processing
texts = data['polarity'].tolist()

# Generate BERT embeddings
batch_embeddings = get_sentence_embeddings_batch(texts, model, batch_size=32)

# Convert embeddings to a list of Python lists
batch_embeddings = [emb.tolist() for emb in batch_embeddings]

# Add embeddings as a new column in the DataFrame
data['bert_embeddings_polarity'] = batch_embeddings


# In[ ]:





# In[10]:


trans_expanded = pd.DataFrame(data['bert_embeddings_trans'].tolist(), index=data.index)
trans_expanded.columns = [f'trans_emb_{i}' for i in range(trans_expanded.shape[1])]


# In[11]:


resume_expanded = pd.DataFrame(data['bert_embeddings_resume'].tolist(), index=data.index)
resume_expanded.columns = [f'resume_emb_{i}' for i in range(resume_expanded.shape[1])]


# In[12]:


jd_expanded = pd.DataFrame(data['bert_embeddings_jd'].tolist(), index=data.index)
jd_expanded.columns = [f'jd_emb_{i}' for i in range(resume_expanded.shape[1])]


# In[13]:


reason_for_decision_expanded = pd.DataFrame(data['bert_embeddings_reason'].tolist(), index=data.index)
reason_for_decision_expanded.columns = [f'reason_emb_{i}' for i in range(resume_expanded.shape[1])]


# In[14]:


polarity_expanded = pd.DataFrame(data['bert_embeddings_polarity'].tolist(), index=data.index)
polarity_expanded.columns = [f'polarity_emb_{i}' for i in range(resume_expanded.shape[1])]


# In[ ]:





# In[15]:


data


# In[ ]:





# Similarity

# In[17]:


# Function to calculate similarity and create bins
def calculate_similarity_with_bins_from_embeddings(resume_embeddings, job_description_embeddings, bins, bin_labels,role, decision):
    """
    Calculate similarity scores using precomputed embeddings and categorize them into bins.

    Args:
        resume_embeddings (numpy.ndarray): Array of resume embeddings.
        job_description_embeddings (numpy.ndarray): Array of job description embeddings.
        bins (list): Bin edges for categorizing similarity scores.
        bin_labels (list): Labels for the bins.

    Returns:
        pd.DataFrame: DataFrame with similarity scores and bin labels.
    """
    # Calculate cosine similarity for each pair of resume and job description
    print("Calculating similarities...")
    similarities = [cosine_similarity([resume], [job_description])[0][0]
                    for resume, job_description in zip(resume_embeddings, job_description_embeddings)]

    # Categorize similarities into bins
    similarity_bins = pd.cut(similarities, bins=bins, labels=bin_labels, include_lowest=True)

    # Create a DataFrame with results
    results = pd.DataFrame({
        "Similarity": similarities,
        "Bin": similarity_bins,
        "Role": role,
        "Decision": decision
    })
    
    return results


# In[18]:


resume_embeddings = data['bert_embeddings_resume']  # Load precomputed resume embeddings
job_description_embeddings = data['bert_embeddings_jd']  # Load precomputed job description embeddings
role = data['Role']
decision = data['decision']


# In[36]:


# Define bins and labels
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

# Calculate similarity and categorize into bins
results = calculate_similarity_with_bins_from_embeddings(
    resume_embeddings, 
    job_description_embeddings, 
    bins=bins, 
    bin_labels=bin_labels,
    role = role,
    decision = decision,
)

data1 = pd.DataFrame(results)


# In[37]:


data1


# In[38]:


data1.to_csv('similarity.csv')


# In[ ]:





# In[39]:


data2 = pd.read_csv('similarity.csv')


# In[40]:


# Define mapping for combining roles
role_mapping = {
    'product manager': 'Product Manager',
    'project manager': 'Product Manager',
    
    'software engineer': 'Software Engineer',
    'software developer': 'Software Engineer',
    'devops engineer': 'Software Engineer',
    'system administrator': 'Software Engineer',
    'game developer': 'Software Engineer',
    'mobile app developer': 'Software Engineer',
    'cloud architect': 'Software Engineer',
    'ai engineer': 'Software Engineer',
    
    'data analyst': 'Data Analyst',
    'business analyst': 'Data Analyst',
    'digital marketing specialist': 'Data Analyst',
    
    'data scientist': 'Data Scientist',
    'machine learning engineer': 'Data Scientist',
    
    'data engineer': 'Data Engineer',
    'database administrator': 'Data Engineer',
    'cloud architect': 'Data Engineer',
    
    'ui designer': 'UI Designer',
    'ui engineer': 'UI Designer',
    'ui / ux designer': 'UI Designer',
    'graphic designer': 'UI Designer',
    
    'cybersecurity specialist': 'Cybersecurity Specialist',
    'content writer': 'Content Writer',
    'hr specialist': 'HR Specialist',
    'network engineer': 'Network Engineer'
}

# Apply the mapping to the dataset
data2['Role'] = data2['Role'].map(role_mapping)

data2


# In[41]:


# Replace 0 with 'Reject' and 1 with 'Accept' in the decision column
data2['Decision'] = data2['Decision'].map({0: 'Reject', 1: 'Accept'})

# Define a function to calculate thresholds for each group (role + decision)
def calculate_threshold(group, method="mean", percentile=None):
    if method == "mean":
        return group["Similarity"].mean()
    elif method == "median":
        return group["Similarity"].median()
    elif method == "percentile" and percentile is not None:
        return np.percentile(group["Similarity"], percentile)
    else:
        raise ValueError("Invalid method or missing percentile for 'percentile' method.")


# In[42]:


# Calculate threshold for each (role, decision) group
thresholds = (
    data2.groupby(["Role", "Decision"])
    .apply(calculate_threshold, method="mean")  # Change method to "median" or "percentile" if needed
    .reset_index(drop=True)
)


# In[43]:


# Merge thresholds with data2 to include the 'Threshold' column
data2['Threshold'] = thresholds


# In[48]:


# Create a table with Role, Decision, Threshold, and Count
result_table = data2.groupby(["Role", "Bin", "Decision"]).size().reset_index(name='Count')


# In[54]:


# Pivot the table so that each bin has separate columns for Accept and Reject counts
pivot_table = result_table.pivot_table(
    index=["Role", "Decision"], 
    columns="Bin", 
    values="Count", 
    aggfunc="sum", 
    fill_value=0
).reset_index()

# Flatten the multi-index in the columns for a cleaner presentation
pivot_table.columns = [f"{col}" for col in pivot_table.columns]

# Reset index to make the table more readable
pivot_table.reset_index(inplace=True)

pivot_table


# In[ ]:





# Prediction

# In[247]:


pred_data = pd.read_excel("Copy of prediction_data.xlsx")


# In[ ]:





# Vectorizer model

# In[248]:


with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


# Sentence_transformer model

# In[249]:


# Load the SentenceTransformer model
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')


# Classifier model

# In[250]:


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

# In[261]:


# Calculate resume and job description similarity (Cosine Similarity)
vectorizer = TfidfVectorizer()
resume_jd_similarity = []
for i in range(len(pred_data)):
    resume = pred_data['Resume'][i]
    jd = pred_data['Job Description'][i]
    similarity = cosine_similarity(tfidf_vectorizer.transform([resume, jd]))[0, 1]
    resume_jd_similarity.append(similarity)
pred_data['resume_jd_similarity'] = resume_jd_similarity


# In[262]:


# Calculate resume and transcript similarity (Cosine Similarity)
resume_transcript_similarity = []
for i in range(len(pred_data)):
    resume = pred_data['Resume'][i]
    transcript = pred_data['Transcript'][i]
    similarity = cosine_similarity(tfidf_vectorizer.transform([resume, transcript]))[0, 1]
    resume_transcript_similarity.append(similarity)
pred_data['resume_transcript_similarity'] = resume_transcript_similarity


# In[263]:


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


# In[264]:


# Function to calculate lexical diversity
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

# Compute lexical diversity for each transcript
pred_data['lexical_diversity'] = pred_data['Transcript'].apply(lexical_diversity)

# Calculate statistics
average_diversity = pred_data['lexical_diversity'].mean()


# In[265]:


# Length of transcript (number of words)
pred_data['transcript_length'] = pred_data['Transcript'].apply(lambda x: len(x.split()))

# Calculate statistics
average_length = pred_data['transcript_length'].mean()
min_length = pred_data['transcript_length'].min()
max_length = pred_data['transcript_length'].max()


# In[266]:


# Function to compute similarity score between Resume and Job Description
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Calculate technical skill matching score
pred_data['technical_skill_match'] = pred_data.apply(lambda row: compute_similarity(row['Resume'], row['Job Description']), axis=1)


# In[267]:


#Soft Skills
pred_data['soft_skills_sentiment'] = pred_data['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[268]:


# Resume length (number of words)
pred_data['resume_length'] = pred_data['Resume'].apply(lambda x: len(x.split()))


# In[269]:


# Job Description Experience Match (Simple matching based on keywords, could be improved)
pred_data['job_description_experience_match'] = pred_data.apply(lambda row: len(set(row['Resume'].split()) & set(row['Job Description'].split())), axis=1)


# In[270]:


#Cultural fit sentiment
pred_data['cultural_fit_sentiment'] = pred_data['Reason for decision'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[271]:


#job score
def job_fit_analysis(job_desc, transcript):
    # You can use similarity or keyword matching here
    job_keywords = job_desc.split()
    transcript_keywords = transcript.split()
    common_keywords = set(job_keywords).intersection(transcript_keywords)
    return len(common_keywords) / len(job_keywords)

pred_data['job_fit_score'] = pred_data.apply(lambda row: job_fit_analysis(row['Job Description'], row['Transcript']), axis=1)


# In[272]:


import re

# Define a function to calculate the confidence score
def calculate_confidence_score(text):
    # Count occurrences of "I think" and "Maybe" (case-insensitive)
    confidence_phrases = re.findall(r'\b(I think|Maybe)\b', text, flags=re.IGNORECASE)
    return len(confidence_phrases)

# Apply the function to calculate confidence scores
pred_data['confidence_score'] = pred_data['Transcript'].apply(calculate_confidence_score)


# In[273]:


#job description complexity
pred_data['job_desc_complexity'] = pred_data['Job Description'].apply(lambda x: textstat.flesch_reading_ease(x))


# In[274]:


#interaction quality
pred_data['interaction_quality'] = pred_data['num_words_in_transcript'] * pred_data['sentiment']


# In[275]:


pred_data['clarity_score'] = pred_data['Transcript'].apply(lambda x: textstat.flesch_reading_ease(x))


# In[276]:


# Text complexity (resume and transcript - using a simple metric like Flesch Reading Ease)
def text_complexity(text):
    # Implement text complexity (e.g., Flesch Reading Ease)
    # Here's a placeholder function:
    return len(text.split()) / len(set(text.split()))  # A basic metric

pred_data['text_complexity_transcript'] = pred_data['Transcript'].apply(text_complexity)
pred_data['text_complexity_resume'] = pred_data['Resume'].apply(text_complexity)


# In[277]:


pred_data.columns


# In[ ]:





# In[ ]:





# In[278]:


# Define a function to convert the text into embeddings
def get_embeddings(texts, model):
    return np.array([model.encode(text) for text in texts])


# In[279]:


# Get embeddings for 'Transcript', 'Job Description', and 'Resume'
transcript_embeddings = get_embeddings(pred_data['Transcript'], sentence_transformer)
job_desc_embeddings = get_embeddings(pred_data['Job Description'], sentence_transformer)
resume_embeddings = get_embeddings(pred_data['Resume'], sentence_transformer)
reason_embeddings = get_embeddings(pred_data['Reason for decision'], sentence_transformer)
polarity_embeddings = get_embeddings(pred_data['polarity'], sentence_transformer)


# In[280]:


print(pred_data.columns.tolist())


# In[281]:


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


# In[282]:


# Predict decision using embedding features
embed_decision = classifier.predict(features)


# In[283]:


print("Predicted decision:", embed_decision)


# In[287]:


# Add predictions to the DataFrame
pred_data['Decision'] = ['Accept' if pred == 1 else 'Reject' for pred in embed_decision]


# In[289]:


pred_data.to_csv('prediction.csv')


# In[ ]:





# Send mail

# In[290]:


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os


# In[303]:


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





# In[ ]:





# In[ ]:




