#!/usr/bin/env python
# coding: utf-8

# In[24]:


import random
import pandas as pd
from together import Together
from faker import Faker
import os
import time


# In[25]:


os.environ["TOGETHER_API_KEY"] = "e89932e7d4565879c28cfbfc20d59ed9811609ea306e136f85078e743e62ea5a"
client = Together()


# In[26]:


roles = {
    "Data Scientist":['Python', 'Data Manipulation', 'Machine Learning', 'Statistics and Probability', 'NLP', 'Deep Learning', 'Data Visualization'],
    "Data Engineer":['Big Data Frameworks', 'Data Pipelines', 'ETL Tools', 'Databases', 'Cloud Platforms', 'Python', 'Data Warehousing', 'CI/CD Tools'],
    "Software Engineer": ['Java', 'Algorithm and Data Structure', 'Web Development', 'Backend Development', 'Version Control', 'DevOps','Testing', 'API'],
    "Product Manager": ['Strategic Planning', 'Roadmap Development','Stakeholder management','Agile and Scrum Methodologies', 'Prototyping', 'Business Communication'],
    "UI Engineer": ['Frontend Frameworks', 'Responsive Design', 'Design Tools', 'Version Control', 'Animation Libraries']
}


# In[27]:


experience_levels = ["Entry-level", "Mid-level", "Senior-level", "Lead", "Director"]
work_environments = ["Remote", "Hybrid", "In-office", "On-Site"]


# In[28]:


select_reasons = [
    "Provided innovative and creative solutions during the interview",
    "Displayed excellent collaboration and teamwork mindset",
    "Showcased the ability to quickly adapt and learn new concepts",
    "Demonstrated a strong understanding of the company’s goals and how they could contribute",
    "Presented clean, efficient, and well-documented code solutions"
]
reject_reasons = [
    "Failed to address performance and scalability concerns in solutions",
    "Had difficulty applying theoretical knowledge to practical scenarios",
    "Showed lack of preparedness for typical interview challenges",
    "Displayed rigidity in adapting to alternative approaches or feedback",
    "Did not demonstrate curiosity or initiative in exploring the problem space"
]


# In[29]:


def get_reasons(outcome, num_reasons=2):
    if outcome == 'select':
        list = random.sample(select_reasons, k=num_reasons)
    elif outcome == 'reject':
        list = random.sample(reject_reasons, k=num_reasons)
    return ', '.join(list)


# In[30]:


def call_together_api(prompt):
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": f"{prompt}"}],
    )
    content = response.choices[0].message.content
    return content.replace("**", "")


# In[31]:


fake = Faker()


# In[32]:


def generate_candidate_data(n):
    data = []
    for _ in range(n):
        id_formatted = f"durgba{_+1:03d}"
        name = fake.name()
        role = random.choice(list(roles.keys()))
        skillset = roles[role]
        poor_skills = random.sample(skillset, random.randint(1, min(3, len(skillset))))
        outcome = random.choice(["select","reject"])

        transcript_prompt = (
            f"Simulate an interview for a candidate named {name} applying for the role of {role}."
            f"The candidate performs poorly in the following skills: {', '.join(poor_skills)}. Generate a professional transcript."
            f"Do not generate generic statements like 'Here is a simulated interview transcript for the ...'."
        )
        transcript = call_together_api(transcript_prompt)

        resume_prompt = (
            f"Create a resume for the candidate named {name} applying for the job role of {role}. "
            f"The candidate is skilled in {', '.join([skill for skill in skillset if skill not in poor_skills])}"
            f"Do not generate generic statements like 'Here's a sample resume for the ...'."
        )
        resume = call_together_api(resume_prompt)

        expected_experience = random.choice(["0-2 years", "3-5 years", "6-8 years", "9+ years"])
        job_description = (
            f"Skilled {role} with expertise in {', '.join(random.sample(skillset,min(4,len(skillset))))}"
            f"\nExpected_experience : {expected_experience}"
        )

        data.append({
            "Id": id_formatted,
            "Name": name,
            "Role": role,
            "Transcript": transcript,
            "Resume": resume,
            "Performance": outcome,
            "Reason for decision": get_reasons(outcome),
            "Job description": job_description
        })

        time.sleep(5)

    return pd.DataFrame(data)


# In[33]:


n = int(input("Enter the number of candidates: "))
data = generate_candidate_data(n)
data


# In[34]:


data.to_excel("candidates_list.xlsx", index=False)


# In[ ]:




