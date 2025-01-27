#!/usr/bin/env python
# coding: utf-8

# PREPROCESSING

# Import Libraries

# In[46]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import textstat
from sklearn.preprocessing import LabelEncoder
import pickle
import os


# In[47]:


# Define the folder containing the Excel files
folder_path = r"D:\Internship - Infosys\Project"

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is an Excel file (e.g., .xlsx or .xls)
    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)
        
        # Create the corresponding CSV file name
        csv_file_name = os.path.splitext(file_name)[0] + ".csv"
        csv_file_path = os.path.join(folder_path, csv_file_name)
        
        # Save the DataFrame as a CSV file
        df.to_csv(csv_file_path, index=False)
        
        print(f"Converted {file_name} to {csv_file_name}")

print("All Excel files have been converted to CSV.")


# In[ ]:





# Check the headers rows with reference header

# In[48]:


# Define the folder containing the CSV files
folder_path = r"D:\Internship - Infosys\Project"

# Reference headers to compare
reference_headers = [
    "ID", "Name", "Role", "Transcript", "Resume", 
    "decision", "Reason for decision", 
    "Job Description", "num_words_in_transcript"
]

# Iterate through each CSV file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a CSV file
    if file_name.endswith(".csv"):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read only the header row
        try:
            with open(file_path, 'r') as file:
                header_row = file.readline().strip().split(',')
            
            # Compare with the reference headers
            if header_row != reference_headers:
                print(f"\nHeader mismatch in file: {file_name}")
                print("File Headers: ", header_row)
                print("Reference Headers: ", reference_headers)
            else:
                print(f"Headers match for file: {file_name}")
        
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

print("\nHeader comparison completed.")


# In[ ]:





# Convert all text to lowercase

# In[49]:


from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def convert_text_to_lowercase_bert(text):
    # Tokenize the text and convert it to lowercase
    tokens = tokenizer.tokenize(text)  # Tokenizing the input text
    lowercased_tokens = [token.lower() for token in tokens]  # Convert tokens to lowercase
    lowercased_text = tokenizer.convert_tokens_to_string(lowercased_tokens)  # Convert tokens back to text
    return lowercased_text

def convert_csv_to_lowercase_using_bert(folder_path):
    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            # Apply the BERT-based lowercase conversion to each string cell
            for col in df.select_dtypes(include=['object']).columns:  # Only process string columns
                df[col] = df[col].apply(convert_text_to_lowercase_bert)

            # Save the modified DataFrame back to the CSV
            df.to_csv(file_path, index=False)
            print(f"File '{file_name}' converted to lowercase using BERT!")

# Specify the folder containing the CSV files
folder_path = "D:\Internship - Infosys\Project"
convert_csv_to_lowercase_using_bert(folder_path)


# Add num_words_in_transcript in the data and find it

# In[50]:


import spacy
import re

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define the folder containing the CSV files
folder_path = r"D:\Internship - Infosys\Project"

# Function to clean and calculate word count using SpaCy tokenizer
def get_word_count(text):
    # Handle empty text or NaN
    if not text or pd.isna(text):
        return 0
    
    # Remove unwanted punctuation like ellipses (...) and other non-word characters
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation like periods, commas, etc.
    
    # Ensure the input is a string and tokenize it
    doc = nlp(str(text))
    
    # Count all non-punctuation, non-whitespace tokens, and include contractions as single words
    word_count = len([token.text for token in doc if token.is_alpha and len(token.text) > 0])
    
    return word_count

# Iterate through each CSV file and process
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Load CSV into DataFrame
            df = pd.read_csv(file_path)
            
            # Ensure 'num_words_in_transcript' column is added
            if "num_words_in_transcript" not in df.columns:
                if "Transcript" in df.columns:
                    # Apply the word count function to each row of the 'Transcript' column
                    df["num_words_in_transcript"] = df["Transcript"].apply(lambda x: get_word_count(x) if pd.notna(x) else 0)
                else:
                    # If 'Transcript' column is missing, add NaN values
                    df["num_words_in_transcript"] = pd.NA

                # Save the updated DataFrame back to the file
                df.to_csv(file_path, index=False)
                print(f"Added 'num_words_in_transcript' column to {file_name}")
            else:
                print(f"'num_words_in_transcript' column already exists in {file_name}")
        
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print("\nProcessing completed.")


# Update the decision column

# In[51]:


def process_decision_column_in_folder(folder_path):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Loop through each CSV file
    for csv_file in csv_files:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, csv_file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Replace 'selected' -> 'select' and 'rejected' -> 'reject' in the 'decision' column
        df['decision'] = df['decision'].replace({'selected': 'select', 'rejected': 'reject'})

        # Save the modified DataFrame back to CSV (overwriting the original or saving as new)
        df.to_csv(file_path, index=False)  # Overwrites the original file
        # Or you can save it as a new file if you prefer:
        # df.to_csv(f'processed_{csv_file}', index=False)

        print(f"Processed {csv_file} and updated 'decision' column.")

# Example usage:
folder_path = 'D:\Internship - Infosys\Project'  # Replace with your actual folder path
process_decision_column_in_folder(folder_path)


# Concatinate all the dataset to a single dataset

# In[52]:


import os
import pandas as pd

def concatenate_datasets_from_folder(folder_path):
    # Initialize an empty list to store the dataframes
    dataframes = []

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Loop through each CSV file
    for csv_file in csv_files:
        # Construct the full path to the file
        file_path = os.path.join(folder_path, csv_file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate all the dataframes into one single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

# Example usage:
folder_path = 'D:\Internship - Infosys\Project'  # Replace with your actual folder path

# Get the combined dataset
final_combined_df = concatenate_datasets_from_folder(folder_path)

# Display the final concatenated DataFrame
print(final_combined_df)
final_combined_df.to_csv('combined_dataset.csv', index=False)


# Change the roles

# In[91]:


data2 = pd.read_csv('combined_dataset.csv')

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

data2.to_csv('combined_dataset.csv')


# In[ ]:





# Descriptive Statistics

# Numerical Features -> num_words_in_transcript

# In[92]:


def load_and_aggregate(file_name):
    aggregated_dfs = []  # List to hold all aggregated DataFrames
    
    # Loop through all files in the folder
    df = pd.read_csv(file_name)  # Read the CSV file
            
            # Perform the aggregation on the DataFrame
    aggregated_df = df[['num_words_in_transcript', 'decision', 'Role']].groupby(['Role', 'decision']).agg(
    mean=('num_words_in_transcript', 'mean'),
    median=('num_words_in_transcript', 'median'),
    std=('num_words_in_transcript', 'std'),
    min=('num_words_in_transcript', 'min'),
    max=('num_words_in_transcript', 'max')
    ).reset_index()
            
    aggregated_dfs.append(aggregated_df)  # Add to the list
    
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    return combined_df

# Usage
file_name = 'combined_dataset.csv'  # Replace with your actual folder path
result = load_and_aggregate(file_name)

# Convert the result to a DataFrame (it already is a DataFrame)
result_df = pd.DataFrame(result)


result_df.to_csv('aggregate.csv')


# In[93]:


result_df


# Categorical Features -> Role

# In[94]:


import pandas as pd
import matplotlib.pyplot as plt

# Function to load data and calculate role frequency distribution
def calculate_role_frequency(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)
    
    # Count the frequency of each role
    role_count = df['Role'].value_counts()
    
    # Convert the series into a DataFrame for easier handling
    role_frequency_df = role_count.reset_index()
    role_frequency_df.columns = ['Role', 'Frequency']
    
    return role_frequency_df

# Usage
file_name = 'combined_dataset.csv'  # Replace with your actual file path
role_frequency_df = calculate_role_frequency(file_name)

# Display the frequency distribution as a table
print(role_frequency_df)

# Plot the frequency distribution
plt.figure(figsize=(10, 6))
role_frequency_df.sort_values(by='Frequency', ascending=False).plot(kind='bar', x='Role', y='Frequency', legend=False)
plt.title('Role Frequency Distribution')
plt.xlabel('Role')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Transcript Sentiment

# In[95]:


import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Function to perform sentiment analysis
def perform_sentiment_analysis(df, transcript_column='Transcript'):
    # Initialize the SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to the 'Transcript' column
    df['Sentiment Score'] = df[transcript_column].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    
    # Categorize the sentiment into Positive, Negative, and Neutral
    df['Sentiment'] = df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    
    return df

# Function to analyze sentiment distribution based on decision
def analyze_sentiment_by_decision(file_name):
    # Read the CSV file
    df = pd.read_csv(file_name)
    
    # Perform sentiment analysis on the Transcript column
    df = perform_sentiment_analysis(df)
    
    # Group by Decision and Sentiment to count occurrences
    sentiment_by_decision = df.groupby(['decision', 'Sentiment']).size().unstack().fillna(0)
    
    return df, sentiment_by_decision

# Usage
file_name = 'combined_dataset.csv'  # Replace with your actual file path
df, sentiment_by_decision = analyze_sentiment_by_decision(file_name)

# Display sentiment analysis result by decision
print(sentiment_by_decision)

# Plot sentiment distribution by decision
sentiment_by_decision.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution by Decision')
plt.xlabel('Decision')
plt.ylabel('Count of Sentiments')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# The vast majority of transcripts are classified as having a positive sentiment, with very few classified as negative sentiment, and no neutral sentiment at all.

# In[ ]:





# In[96]:


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    # Remove stop words, convert to lowercase, and remove non-alphabetic characters
    return ' '.join(word.lower() for word in str(text).split() if word.lower() not in stop_words and word.isalpha())

# Function to generate word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

# Function to create word clouds for selected and rejected candidates
def create_word_clouds(file_name):
    # Load data
    df = pd.read_csv(file_name)

    # Preprocess text columns
    df['Transcript Cleaned'] = df['Transcript'].apply(preprocess_text)
    df['Resume Cleaned'] = df['Resume'].apply(preprocess_text)

    # Split data by decision
    selected = df[df['decision'] == 'select']
    rejected = df[df['decision'] == 'reject']

    # Generate word clouds for each group and column
    for column, title_prefix in [('Transcript Cleaned', 'Transcript'), ('Resume Cleaned', 'Resume')]:
        generate_word_cloud(' '.join(selected[column]), f'{title_prefix} Word Cloud (Selected Candidates)')
        generate_word_cloud(' '.join(rejected[column]), f'{title_prefix} Word Cloud (Rejected Candidates)')

# Usage
file_name = 'combined_dataset.csv'  # Replace with the path to your dataset
create_word_clouds(file_name)


# The transcript of selected students has the words like {experience, previous role, skills, tools, problem solving and the words that are more related to the job }
# The transcript of rejected students has only the normal verbal words which are not related to industry. This also shows that the candidate only knows about the title of job role and not much about it
# In Resume, the selected candidate has given more skillsets in the resume than the rejected candidate 

# In[ ]:





# In[97]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('combined_dataset.csv')  # Replace with your dataset path

# Function to calculate cultural fit score
def calculate_cultural_fit(df):
    # Extract relevant columns
    transcripts = df['Transcript'].fillna('')  # Candidate responses
    job_descriptions = df['Job Description'].fillna('')  # Role expectations

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Combine both columns for TF-IDF training
    all_texts = pd.concat([transcripts, job_descriptions], axis=0)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Split back into transcripts and job descriptions
    transcripts_tfidf = tfidf_matrix[:len(transcripts)]
    job_descriptions_tfidf = tfidf_matrix[len(transcripts):]

    # Calculate cosine similarity between transcripts and job descriptions
    similarity_scores = []
    for i in range(len(transcripts)):
        similarity = cosine_similarity(transcripts_tfidf[i], job_descriptions_tfidf[i])
        similarity_scores.append(similarity[0][0])

    # Add the scores to the DataFrame
    df['Cultural_Fit_Score'] = similarity_scores

    return df

# Calculate the cultural fit score
df_with_scores = calculate_cultural_fit(df)

# Save the result
df_with_scores.to_csv('cultural_fit_scores.csv', index=False)

# Print sample output
print(df_with_scores[['Role', 'Cultural_Fit_Score', 'decision']].head())


# In[98]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the cultural fit score for selected vs rejected candidates
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_with_scores, x='decision', y='Cultural_Fit_Score', palette='Set2')
plt.title('Cultural Fit Score Distribution by Decision (Select vs Reject)')
plt.xlabel('Decision')
plt.ylabel('Cultural Fit Score')
plt.tight_layout()
plt.show()


# In[99]:


# Group by decision and calculate mean and median of cultural fit scores
stats = df_with_scores.groupby('decision')['Cultural_Fit_Score'].agg(['mean', 'median'])
print(stats)


# In[100]:


from scipy.stats import ttest_ind

# Separate the scores for "select" and "reject" candidates
select_scores = df_with_scores[df_with_scores['decision'] == 'select']['Cultural_Fit_Score']
reject_scores = df_with_scores[df_with_scores['decision'] == 'reject']['Cultural_Fit_Score']

# Perform the t-test
t_stat, p_value = ttest_ind(select_scores, reject_scores)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# The difference in cultural fit scores between candidates who are selected and those who are rejected is statistically significant. Since the p-value is very small (much less than the typical threshold of 0.05), we can reject the null hypothesis.  A higher t-statistic suggests a larger difference between the two groups.

# In[ ]:





# In[101]:


import pandas as pd


df = pd.read_csv('cultural_fit_scores.csv')

# Step 1: Group by 'Role' and 'decision', and calculate the min, mean, median of the 'Cultural Fit Score'
grouped_df = df.groupby(['Role', 'decision'])['Cultural_Fit_Score'].agg(['min', 'mean', 'median'])

# Step 2: Reset index to convert multi-level indices into columns
grouped_df = grouped_df.reset_index()

# Step 3: Pivot the table so that 'decision' values become separate columns
pivot_df = grouped_df.pivot_table(index='Role', columns='decision', values=['min', 'mean', 'median'])

# Step 4: Flatten the multi-level columns resulting from pivot
pivot_df.columns = [f'{stat}_{decision}' for stat, decision in pivot_df.columns]

# Step 5: Print the final DataFrame
data = pd.DataFrame(pivot_df)
data


# Roles like data analyst and software engineer show smaller mean differences between selected and rejected candidates, suggesting that candidates with varying levels of Cultural Fit can still be selected.
# For most roles, the mean is higher than the median, which might indicate a skewed distribution
# The minimum for selected candidates in data analyst role is extremely low (0.003219), suggesting that even candidates with very low Cultural Fit Scores are sometimes selected.
# Higher Cultural Fit Scores seem to play a stronger role in selection decisions for certain roles

# In[102]:


combined_dataset = pd.read_csv('cultural_fit_scores.csv')
# Step 1: Calculate the length of Transcript and Resume
combined_dataset['Transcript Length'] = combined_dataset['Transcript'].apply(lambda x: len(x.split()))
combined_dataset['Resume Length'] = combined_dataset['Resume'].apply(lambda x: len(x.split()))

# Step 2: Calculate the Transcript Length to Resume Length Ratio
combined_dataset['Transcript to Resume Length Ratio'] = combined_dataset['Transcript Length'] / combined_dataset['Resume Length']

# Step 3: Calculate Cultural Fit Score to Transcript/Resume Length Ratio
combined_dataset['Cultural Fit to Transcript Length Ratio'] = combined_dataset['Cultural_Fit_Score'] / combined_dataset['Transcript Length']
combined_dataset['Cultural Fit to Resume Length Ratio'] = combined_dataset['Cultural_Fit_Score'] / combined_dataset['Resume Length']

# Step 4: Correlation with Selection Decision (selected = 1, rejected = 0)
# Assuming 'decision' is the column with selection results (1 = selected, 0 = rejected)
decision_map = {'select': 1, 'reject': 0}
combined_dataset['decision_numeric'] = combined_dataset['decision'].map(decision_map)

# Step 5: Calculate correlation for the ratios with the decision
correlations = combined_dataset[['Transcript to Resume Length Ratio', 'Cultural Fit to Transcript Length Ratio', 'Cultural Fit to Resume Length Ratio', 'decision_numeric']].corr()

# Display the correlations
data = pd.DataFrame(correlations)
data


# The correlations with decision_numeric are all weak, suggesting that Cultural Fit relative to length and Transcript to Resume Length Ratio have a minor role in the final decision-making process.

# In[ ]:





# In[ ]:





# FEATURE EXTRACTION

# In[103]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import textstat
from sklearn.preprocessing import LabelEncoder
import pickle


# In[104]:


data = pd.read_csv('combined_dataset.csv')
data


# In[105]:


data.info()
data.describe()


# In[106]:


# Checking for missing values
data.isnull().sum()


# In[107]:


# Calculate resume and job description similarity (Cosine Similarity)
vectorizer = TfidfVectorizer()
resume_jd_similarity = []
for i in range(len(data)):
    resume = data['Resume'][i]
    jd = data['Job Description'][i]
    similarity = cosine_similarity(vectorizer.fit_transform([resume, jd]))[0, 1]
    resume_jd_similarity.append(similarity)
data['resume_jd_similarity'] = resume_jd_similarity


# In[108]:


# Calculate resume and transcript similarity (Cosine Similarity)
resume_transcript_similarity = []
for i in range(len(data)):
    resume = data['Resume'][i]
    transcript = data['Transcript'][i]
    similarity = cosine_similarity(vectorizer.fit_transform([resume, transcript]))[0, 1]
    resume_transcript_similarity.append(similarity)
data['resume_transcript_similarity'] = resume_transcript_similarity


# In[109]:


# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Lists to store sentiment results
sentiments = []
polarity = []

# Perform sentiment analysis on each transcript
for i in range(len(data)):
    transcript = data['Transcript'][i]
    sentiment_score = sia.polarity_scores(transcript)
    sentiments.append(sentiment_score['compound'])  # Compound sentiment score
    polarity.append('positive' if sentiment_score['compound'] > 0 
                    else 'negative' if sentiment_score['compound'] < 0 
                    else 'neutral')

# Add the results to the DataFrame
data['sentiment'] = sentiments
data['polarity'] = polarity

# Count of each sentiment category
polarity_counts = data['polarity'].value_counts()
print("\n2. Sentiment Polarity Distribution:")
for polarity, count in polarity_counts.items():
    print(f"   - {polarity.capitalize()}: {count} occurrences ({(count / len(data) * 100):.2f}%)")

# Overall average sentiment score
average_sentiment = data['sentiment'].mean()
print(f"\n3. Overall Average Sentiment Score: {average_sentiment:.2f}")
if average_sentiment > 0:
    print("   - The overall sentiment of the transcripts is positive.")
elif average_sentiment < 0:
    print("   - The overall sentiment of the transcripts is negative.")
else:
    print("   - The overall sentiment of the transcripts is neutral.")


# In[ ]:





# In[110]:


# Function to calculate lexical diversity
def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words)

# Compute lexical diversity for each transcript
data['lexical_diversity'] = data['Transcript'].apply(lexical_diversity)

# Calculate statistics
average_diversity = data['lexical_diversity'].mean()


# In[111]:


# Length of transcript (number of words)
data['transcript_length'] = data['Transcript'].apply(lambda x: len(x.split()))

# Calculate statistics
average_length = data['transcript_length'].mean()
min_length = data['transcript_length'].min()
max_length = data['transcript_length'].max()


# In[112]:


# Encoding the target variable (select/reject)
le = LabelEncoder()
data['decision'] = le.fit_transform(data['decision'])  # 0: reject, 1: select


# In[113]:


# Function to compute similarity score between Resume and Job Description
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Calculate technical skill matching score
data['technical_skill_match'] = data.apply(lambda row: compute_similarity(row['Resume'], row['Job Description']), axis=1)


# In[114]:


#Soft Skills
from textblob import TextBlob

data['soft_skills_sentiment'] = data['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[115]:


# Resume length (number of words)
data['resume_length'] = data['Resume'].apply(lambda x: len(x.split()))


# In[116]:


# Job Description Experience Match (Simple matching based on keywords, could be improved)
data['job_description_experience_match'] = data.apply(lambda row: len(set(row['Resume'].split()) & set(row['Job Description'].split())), axis=1)


# In[117]:


#Cultural fit sentiment
data['cultural_fit_sentiment'] = data['Reason for decision'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[118]:


#job score
def job_fit_analysis(job_desc, transcript):
    # You can use similarity or keyword matching here
    job_keywords = job_desc.split()
    transcript_keywords = transcript.split()
    common_keywords = set(job_keywords).intersection(transcript_keywords)
    return len(common_keywords) / len(job_keywords)

data['job_fit_score'] = data.apply(lambda row: job_fit_analysis(row['Job Description'], row['Transcript']), axis=1)


# In[119]:


import re

# Define a function to calculate the confidence score
def calculate_confidence_score(text):
    # Count occurrences of "I think" and "Maybe" (case-insensitive)
    confidence_phrases = re.findall(r'\b(I think|Maybe)\b', text, flags=re.IGNORECASE)
    return len(confidence_phrases)

# Apply the function to calculate confidence scores
data['confidence_score'] = data['Transcript'].apply(calculate_confidence_score)


# In[120]:


#job description complexity
data['job_desc_complexity'] = data['Job Description'].apply(lambda x: textstat.flesch_reading_ease(x))


# In[121]:


#interaction quality
data['interaction_quality'] = data['num_words_in_transcript'] * data['sentiment']


# In[122]:


data['clarity_score'] = data['Transcript'].apply(lambda x: textstat.flesch_reading_ease(x))


# In[123]:


# Text complexity (resume and transcript - using a simple metric like Flesch Reading Ease)
def text_complexity(text):
    # Implement text complexity (e.g., Flesch Reading Ease)
    # Here's a placeholder function:
    return len(text.split()) / len(set(text.split()))  # A basic metric

data['text_complexity_transcript'] = data['Transcript'].apply(text_complexity)
data['text_complexity_resume'] = data['Resume'].apply(text_complexity)


# In[124]:


data.to_csv('new_combined_data.csv')


# In[125]:


with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


# In[126]:


data


# In[ ]:





# In[127]:


data = pd.read_csv('new_combined_data.csv')


# In[128]:


# List of numerical features to analyze
numerical_features = ['num_words_in_transcript',
       'resume_jd_similarity', 'resume_transcript_similarity', 'sentiment',
       'transcript_length', 'resume_length',
       'job_description_experience_match', 'text_complexity_transcript',
       'text_complexity_resume', 'lexical_diversity',
       'technical_skill_match', 'soft_skills_sentiment',
       'cultural_fit_sentiment', 'job_fit_score', 'confidence_score',
       'clarity_score', 'job_desc_complexity', 'interaction_quality']

# Loop through each feature and plot its distribution
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True, bins=20)  # Create histogram with KDE curve
    plt.title(f'Distribution of {feature}')
    plt.show()
    
    # Summary of the plot
    print(f"--- Summary of {feature} Distribution ---")
    # Analyze the feature's distribution
    feature_data = data[feature]
    mean_value = feature_data.mean()
    median_value = feature_data.median()
    std_value = feature_data.std()
    
    print(f"   - Mean: {mean_value:.2f}")
    print(f"   - Median: {median_value:.2f}")
    print(f"   - Standard Deviation: {std_value:.2f}")
    
    # Insights based on distribution (adjust based on the feature type)
    if mean_value > median_value:
        print(f"   - The distribution of {feature} is positively skewed.")
    elif mean_value < median_value:
        print(f"   - The distribution of {feature} is negatively skewed.")
    else:
        print(f"   - The distribution of {feature} is symmetric.")
    
    print(f"   - {feature} appears to have a {'normal' if abs(mean_value - median_value) < 0.1 else 'skewed'} distribution.")
    print("   ----------------------------------------------------\n")

