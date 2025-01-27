#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from textblob import TextBlob
import textstat


# In[2]:


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


polarity_expanded = pd.DataFrame(data['bert_embeddings_polarity'].tolist(), index=data.index)
polarity_expanded.columns = [f'polarity_emb_{i}' for i in range(resume_expanded.shape[1])]


# In[14]:


data


# In[ ]:





# Similarity

# In[15]:


resume_embeddings = data['bert_embeddings_resume']  # Load precomputed resume embeddings
job_description_embeddings = data['bert_embeddings_jd']  # Load precomputed job description embeddings


# In[16]:


# Define bins and labels
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


# Calculate similarity for each pair of resume and job description embeddings
similarities = [cosine_similarity([resume], [job_description])[0][0]
                for resume, job_description in zip(resume_embeddings, job_description_embeddings)]

# Assign the similarities to the new column in the data DataFrame
data['similarity'] = similarities

 # Categorize similarities into bins
similarity_bins = pd.cut(similarities, bins=bins, labels=bin_labels, include_lowest=True)

data['similarity_bin'] = similarity_bins

data


# Acceptance Rate

# In[17]:


# Calculate acceptance rate
data['acceptance_rate'] = data.groupby(['similarity_bin', 'Role'], observed=True)['decision'].transform(
    lambda x: x.mean()
)


# Resume Screening Score

# In[18]:


data['resume_screening_score'] = (
    0.5 * data['resume_jd_similarity'] + 
    0.3 * data['cultural_fit_sentiment'] +
    0.2 * data['clarity_score']
)


# Interview Performance Score

# In[19]:


data['interview_performance_score'] = (
    0.3 * data['sentiment'] + 
    0.3 * data['cultural_fit_sentiment'] +
    0.2 * data['soft_skills_sentiment'] +
    0.2 * data['clarity_score']
)


# In[20]:


threshold = 0.7
filtered_data = data[data['resume_screening_score'] >= threshold]


# In[21]:


filtered_data


# In[22]:


data['final_score'] = 0.6 * data['resume_screening_score'] + 0.4 * data['interview_performance_score']


# In[23]:


data


# In[24]:


data.to_csv('similarity.csv')

