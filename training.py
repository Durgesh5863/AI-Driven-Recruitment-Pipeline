#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score


# In[60]:


data = pd.read_csv('new_combined_data.csv')


# In[61]:


#Select features for model training (removed the removed features)
X = data[[
    'num_words_in_transcript',
    'resume_jd_similarity', 'resume_transcript_similarity', 'sentiment',
    'transcript_length', 'resume_length',
    'job_description_experience_match', 'text_complexity_transcript',
    'text_complexity_resume', 'lexical_diversity', 
    'technical_skill_match', 'soft_skills_sentiment',
    'cultural_fit_sentiment', 'job_fit_score', 'confidence_score',
    'clarity_score', 'job_desc_complexity', 'interaction_quality'
]]

y = data['decision']


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression

# In[63]:


from sklearn.linear_model import LogisticRegression

# Hyperparameter tuning: Try different values for regularization strength (C)
param_grid = {'C': [0.1, 0.5, 1, 5, 10]}  # Inverse of regularization strength
log_reg = LogisticRegression(solver='liblinear')

# Use GridSearchCV to find the best regularization parameter C
grid_search = GridSearchCV(log_reg, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Best hyperparameter and model
best_C = grid_search.best_params_['C']
best_log_reg_model = grid_search.best_estimator_

# Predict on test set
y_pred_prob = best_log_reg_model.predict_proba(X_test)[:, 1]
y_pred_binary = best_log_reg_model.predict(X_test)

# Calculate accuracy and ROC AUC
log_reg_accuracy = accuracy_score(y_test, y_pred_binary)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
log_reg_roc_auc = auc(fpr, tpr)

# Output the best model's parameters and performance
print(f"Best Logistic Regression Accuracy: {log_reg_accuracy*100:.4f}")
print(f"Best Logistic Regression AUC: {log_reg_roc_auc:.4f}")


# In[64]:


# --- Error Analysis ---
error_analysis = X_test.copy()
error_analysis['True Label'] = y_test
error_analysis['Predicted Label'] = y_pred_binary
error_analysis['Error'] = error_analysis['True Label'] != error_analysis['Predicted Label']

# Display misclassified samples
misclassified_samples = error_analysis[error_analysis['Error'] == True]
print("Misclassified Instances:")
print(misclassified_samples.head())  # Displaying the first few misclassified instances


# In[65]:


# Counting the total number of misclassified instances
misclassified_count = len(misclassified_samples)

# Professional phrasing
total_misclassified = f"Total number of misclassified instances: {misclassified_count}"
total_misclassified


# In[66]:


# Fit the logistic regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# Now you can calculate the impact as you were doing
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg_model.coef_[0],  # Coefficients from the model
    'Impact': abs(log_reg_model.coef_[0]) * X_train.std()  # Impact = coefficient * std deviation
})

# Sorting by impact to see the features with the highest influence
coefficients_sorted = coefficients.sort_values(by='Impact', ascending=False)

# Printing the impact analysis summary
print("Impact Analysis (Logistic Regression):")
print(coefficients_sorted)


# In[67]:


# --- Feature Importance Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Impact', y='Feature', data=coefficients_sorted)
plt.title('Logistic Regression Feature Importance (Impact Analysis)')
plt.show()

# --- Summary ---
top_features = coefficients_sorted.head(5)  # Top 5 features by impact

# Summarizing the plot
print("Summary of Feature Importance:")
print(f"The plot above shows the importance of features in predicting the target variable based on their impact. The impact is calculated as the product of the absolute coefficient and the standard deviation of each feature.")
print("\nTop 5 most impactful features:")
for i, row in top_features.iterrows():
    print(f"{row['Feature']}: Coefficient = {row['Coefficient']:.4f}, Impact = {row['Impact']:.4f}")


# In[ ]:





# Decision Tree

# In[68]:


from sklearn.tree import DecisionTreeClassifier

# Decision Tree Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

dt_best = grid_search.best_estimator_
dt_y_pred = dt_best.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_y_pred)
dt_roc_auc = roc_auc_score(y_test, dt_best.predict_proba(X_test)[:, 1])


# In[69]:


print(f'Decision Tree Accuracy: {dt_accuracy * 100:.2f}%')
print(f'Decision Tree ROC AUC: {dt_roc_auc:.4f}')


# In[70]:


# Error Analysis
dt_errors = X_test.copy()
dt_errors['True Label'] = y_test
dt_errors['Predicted Label'] = dt_best.predict(X_test)
dt_errors['Error'] = dt_errors['True Label'] != dt_errors['Predicted Label']

# Display a few misclassified instances
misclassified_dt = dt_errors[dt_errors['Error'] == True]
print("Misclassified Instances (Decision Tree):")
print(misclassified_dt.head())


# In[71]:


# Counting the total number of misclassified instances
misclassified_count = len(misclassified_dt)

# Professional phrasing
total_misclassified = f"Total number of misclassified instances: {misclassified_count}"
total_misclassified


# In[72]:


# Impact Analysis (using feature importances)
dt_feature_importance = dt_best.feature_importances_
dt_impact_analysis = pd.DataFrame({
   'Feature': X_train.columns,
   'Importance': dt_feature_importance
}).sort_values(by='Importance', ascending=False)

# Displaying impact
print("Impact Analysis (Decision Tree):")
print(dt_impact_analysis)


# In[73]:


# --- Decision Tree Feature Importance Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=dt_impact_analysis)
plt.title('Decision Tree Feature Importance')
plt.show()

# --- Summary ---
top_dt_features = dt_impact_analysis.head(5)  # Top 5 features by importance

# Summarizing the plot
print("Summary of Feature Importance (Decision Tree):")
print(f"The plot above shows the feature importance in the Decision Tree model. Features with higher importance have a stronger influence on the model's predictions.")
print("\nTop 5 most impactful features:")
for i, row in top_dt_features.iterrows():
    print(f"{row['Feature']}: Importance = {row['Importance']:.4f}")


# In[ ]:





# Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

rf_best = grid_search.best_estimator_
rf_y_pred = rf_best.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_roc_auc = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])

print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')
print(f'Random Forest ROC AUC: {rf_roc_auc:.4f}')


# In[75]:


# Error Analysis
rf_errors = X_test.copy()
rf_errors['True Label'] = y_test
rf_errors['Predicted Label'] = rf_best.predict(X_test)
rf_errors['Error'] = rf_errors['True Label'] != rf_errors['Predicted Label']

# Display a few misclassified instances
misclassified_rf = rf_errors[rf_errors['Error'] == True]
print("Misclassified Instances (Random Forest):")
print(misclassified_rf.head())


# In[76]:


# Counting the total number of misclassified instances
misclassified_count = len(misclassified_rf)

# Professional phrasing
total_misclassified = f'Total number of misclassified instances: {misclassified_count}'
total_misclassified


# In[77]:


# Impact Analysis (using feature importances)
rf_feature_importance = rf_best.feature_importances_
rf_impact_analysis = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_feature_importance
}).sort_values(by='Importance', ascending=False)

# Displaying impact
print("Impact Analysis (Random Forest):")
print(rf_impact_analysis)


# In[78]:


# --- Random Forest Feature Importance Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_impact_analysis)
plt.title('Random Forest Feature Importance')
plt.show()

# --- Summary ---
top_rf_features = rf_impact_analysis.head(5)  # Top 5 features by importance

# Summarizing the plot
print("Summary of Feature Importance (Random Forest):")
print(f"The plot above shows the feature importance in the Random Forest model. Features with higher importance have a stronger influence on the model's predictions.")
print("\nTop 5 most impactful features:")
for i, row in top_rf_features.iterrows():
    print(f"{row['Feature']}: Importance = {row['Importance']:.4f}")


# In[ ]:





# XGBoost

# In[79]:


from xgboost import XGBClassifier

# XGBoost Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 150]
}

xgb = XGBClassifier(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

xgb_best = grid_search.best_estimator_
xgb_y_pred = xgb_best.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
xgb_roc_auc = roc_auc_score(y_test, xgb_best.predict_proba(X_test)[:, 1])

print(f'XGBoost Accuracy: {xgb_accuracy * 100:.2f}%')
print(f'XGBoost ROC AUC: {xgb_roc_auc:.4f}')


# In[80]:


# Error Analysis
xgb_errors = X_test.copy()
xgb_errors['True Label'] = y_test
xgb_errors['Predicted Label'] = xgb_best.predict(X_test)
xgb_errors['Error'] = xgb_errors['True Label'] != xgb_errors['Predicted Label']

# Display a few misclassified instances
misclassified_xgb = xgb_errors[xgb_errors['Error'] == True]
print("Misclassified Instances (XGBoost):")
print(misclassified_xgb.head())


# In[81]:


# Counting the total number of misclassified instances
misclassified_count = len(misclassified_xgb)

# Professional phrasing
total_misclassified = f"Total number of misclassified instances: {misclassified_count}"
total_misclassified


# In[82]:


# Impact Analysis (using feature importances)
xgb_feature_importance = xgb_best.get_booster().get_score(importance_type='weight')
xgb_impact_analysis = pd.DataFrame(list(xgb_feature_importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

# Displaying impact
print("Impact Analysis (XGBoost):")
print(xgb_impact_analysis)


# In[83]:


# --- XGBoost Feature Importance Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_impact_analysis)
plt.title('XGBoost Feature Importance')
plt.show()

# --- Summary ---
top_xgb_features = xgb_impact_analysis.head(5)  # Top 5 features by importance

# Summarizing the plot
print("Summary of Feature Importance (XGBoost):")
print(f"The plot above shows the feature importance in the XGBoost model. Features with higher importance have a stronger influence on the model's predictions.")
print("\nTop 5 most impactful features:")
for i, row in top_xgb_features.iterrows():
    print(f"{row['Feature']}: Importance = {row['Importance']:.4f}")


# In[ ]:





# Save the best model

# In[84]:


import pickle

# Save the trained model to a pickle file
with open('xgb_best_model.pkl', 'wb') as file:
    pickle.dump(xgb_best, file)


# In[ ]:





# In[ ]:





# XG Boost is my best model analysis

# In[85]:


import shap

# Ensure you use the correct trained model, e.g., xgb_best from your tuning process
explainer = shap.Explainer(xgb_best, X_train)

# Calculate SHAP values for the training data
shap_values = explainer(X_train)

# Plot feature importance using SHAP
shap.summary_plot(shap_values, X_train)


# In[86]:


# Beeswarm plot for SHAP values
shap.plots.beeswarm(shap_values)


# Feature-Wise Insights (Beeswarm)
# 
# 1. Cultural Fit Sentiment:
# 
#     High values (red) positively influence predictions, significantly pushing the model output higher.
# 
#     Low values (blue) have a neutral or slightly negative impact, indicating cultural fit is a critical factor.
# 
# 2. Clarity Score:
# 
#     High values strongly increase the output, highlighting that clarity is a significant determinant.
#     
#     Low clarity creates a substantial downward pull on predictions.
# 
# 3. Resume-Transcript Similarity:
# 
#     Higher similarity contributes positively to predictions, as alignment between resume and transcript is key.
# 
#     Lower similarity has an adverse effect, decreasing the model’s confidence in the prediction.
# 
# 4. Resume-JD Similarity:
# 
#     Similar trends as Resume-Transcript Similarity, showing that resume alignment with job descriptions is critical.
# 
# 5. Text Complexity in Transcript:
# 
#     Moderate complexity (middle SHAP values) positively influences predictions.
#     
#     Extremely high complexity negatively impacts output, suggesting diminishing returns or misalignment for overly complex text.
# 
# 6. Transcript Length:
# 
#     Optimal length (moderate values) increases predictions.
# 
#     Too short or excessively long transcripts pull predictions downward, possibly due to insufficient or verbose communication.
# 
# 7. Technical Skill Match:
# 
#     Strong positive influence for high values, reflecting its critical role in decision-making.
# 
#     Low scores drastically reduce predictions.
# 
# 8. Soft Skills Sentiment:
# 
#     High values significantly boost predictions, indicating that soft skills sentiment carries weight.
#     
#     Low values reduce predictions, showing a penalty for poor sentiment.
# 
# 9. Job Fit Score:
# 
#     Strong correlation with positive predictions when high, signifying its core importance.
# 
# 10. Sum of Other Features:
# 
#     Aggregated minor features show mixed effects but have less overall importance compared to primary features.
# 

# In[ ]:





# In[87]:


base_value  = explainer.expected_value
print(f"Base Value: {base_value}")


# In[88]:


shap_values_test = explainer(X_test)  


# In[89]:


# Function to find instances based on prediction type
def find_instance_index(predictions, target):
    if target == "low":
        return np.argmin(predictions)  # Instance with the lowest prediction
    elif target == "high":
        return np.argmax(predictions)  # Instance with the highest prediction
    elif target == "medium":
        median_value = np.median(predictions)
        return np.argmin(np.abs(predictions - median_value))  # Closest to the median

# Get predictions from SHAP base + contributions
predictions = [shap_value.base_values + shap_value.values.sum() for shap_value in shap_values_test]

# Find indices for low, high, and medium predictions
low_index = find_instance_index(predictions, "low")
high_index = find_instance_index(predictions, "high")
medium_index = find_instance_index(predictions, "medium")

# Analyze these indices
for instance_index, target in zip([low_index, high_index, medium_index], ["low", "high", "medium"]):
    shap_value = shap_values_test[instance_index]
    
    # Generate the waterfall plot
    print(f"\n--- SHAP Waterfall Plot for Instance ({target.upper()} prediction) ---")
    shap.plots.waterfall(shap_value)
    
    # Extract information for summary
    feature_contributions = shap_value.values
    base_value = shap_value.base_values 
    predicted_value = base_value + feature_contributions.sum() 
    feature_names = shap_value.feature_names
    top_features = sorted(zip(feature_names, feature_contributions), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    # Convert log-odds to probabilities
    base_probability = 1 / (1 + np.exp(-base_value))  
    predicted_probability = 1 / (1 + np.exp(-predicted_value)) 
    
    # Print summary
    print("\nSummary:")
    print(f"Base Probability: {base_probability:.4%}")
    print(f"Predicted Probability: {predicted_probability:.4%}")
    
    print("\nTop Contributing Features:")
    for feature, contribution in top_features:
        direction = "increases" if contribution > 0 else "decreases"
        print(f"  - {feature}: {contribution:.4f} ({direction} prediction)")
    
    # Feature value insight
    print("\nFeature Value Insights:")
    for feature, contribution in top_features:
        feature_index = feature_names.index(feature)
        feature_value = shap_value.data[feature_index]
        print(f"  - {feature} has a value of {feature_value:.4f}, contributing {contribution:.4f}.")


# In[ ]:





# SHAP Dependency Plot

# In[90]:


# Dependency Plot Feature 1
features_to_plot = "cultural_fit_sentiment" 
shap.dependence_plot(
    ind=features_to_plot, 
    shap_values=shap_values_test.values,  
    features=X_test.values,  
    feature_names=X_test.columns, 
    interaction_index=None, 
    cmap=plt.cm.Reds  
)


# Higher Cultural Fit Sentiment: Higher values of "cultural_fit_sentiment" generally lead to higher SHAP values, suggesting a stronger positive impact on the model's output.
# 
# Clustered Data: Most observations have moderate cultural fit sentiment and SHAP values, indicating that extreme values are less common.
# 
# Outliers Impact: Extreme values of cultural fit sentiment can significantly influence the model's predictions, highlighting the importance of considering outliers in the analysis.

# In[91]:


# Dependency Plot Feature 2
feature_name = "clarity_score"  

shap.dependence_plot(
    ind=feature_name, 
    shap_values=shap_values_test.values,  
    features=X_test.values,  
    feature_names=X_test.columns, 
    interaction_index=None, 
    cmap=plt.cm.Reds  
)


# Negative Impact: Higher clarity_score negatively influences the model's output, especially when it increases beyond 60.
# 
# Clustered Sensitivity: The model shows sensitivity to changes in clarity_score within specific ranges (around 50, 60, and 70).
# 
# Extremes: Extreme values of clarity scores (both low and high) have significant impacts on the model's predictions, with low scores having a positive impact and high scores having a negative impact.
# 
# There is a noticeable drop in SHAP values as clarity_score increases from 40 to 70, after which the SHAP values stabilize around -3 to -4. This trend indicates that the negative impact of clarity scores on the model's prediction becomes more pronounced until it stabilizes.

# In[92]:


# Dependency Plot for Feature 3
feature_name = "resume_transcript_similarity"  

shap.dependence_plot(
    ind=feature_name, 
    shap_values=shap_values_test.values,  
    features=X_test.values, 
    feature_names=X_test.columns,  
    interaction_index=None,  
    cmap=plt.cm.Reds  
)


# Higher Resume-Transcript Similarity: Higher values of "resume_transcript_similarity" generally lead to higher SHAP values, suggesting a stronger positive impact on the model's output.
# 
# Clustered Data: Most observations have moderate resume-transcript similarity and SHAP values, indicating that extreme values are less common.
# 
# Outliers Impact: Extreme values of resume-transcript similarity can significantly influence the model's predictions, highlighting the importance of considering outliers in the analysis.
# 
# The positive correlation between "resume_transcript_similarity" and "SHAP value for resume_transcript_similarity" can be quantified using the correlation coefficient. A high positive correlation coefficient (close to +1) indicates a strong linear relationship.
# 
# A linear regression model can be fitted to the data to quantify the relationship between "resume_transcript_similarity" and SHAP values. The slope of the regression line indicates the rate of increase in SHAP value for a unit increase in resume-transcript similarity.

# In[ ]:





# Partial Dependence Plot

# In[93]:


from sklearn.inspection import PartialDependenceDisplay

#PartialDependence Plot for Feature
feature_name = "transcript_length"
feature_index = X_test.columns.get_loc(feature_name)

PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_test, 
    features=[feature_index], 
    feature_names=X_test.columns,  
    grid_resolution=50,  
    kind="average" 
)
plt.title(f"Partial Dependence Plot for {feature_name}")
plt.show()


# Non-linear Relationship:
# 
# The relationship between transcript length and the predicted outcome is non-linear, with significant fluctuations in the middle range (800 to 1000).
# 
# Optimal Range:
# 
# The range of 700 to 800 appears to be optimal for increasing the predicted outcome, while further increases beyond 1000 show diminishing returns.
# 
# Impact of Shorter and Longer Transcripts:
# 
# Shorter transcripts (around 700) have a lower impact on the predicted outcome, while longer transcripts (beyond 1000) stabilize and do not significantly affect the predictions.
# 
# Trend:
# 
# The overall trend shows an initial increase in partial dependence with increasing transcript length, followed by fluctuations and eventual stabilization.

# 

# In[94]:


#PartialDependence Plot for Feature2
feature_name = "job_desc_complexity"
feature_index = X_test.columns.get_loc(feature_name)

PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_test,  
    features=[feature_index], 
    feature_names=X_test.columns,  
    grid_resolution=50,
    kind="average"  
)
plt.title(f"Partial Dependence Plot for {feature_name}")
plt.show()


# Negative Impact Zone: When "job_desc_complexity" values are between -20 and 0, the feature has a negative impact on the target variable.
# 
# Positive Impact Zone: Values above 0 generally show a positive impact on the target variable, with higher "job_desc_complexity" leading to an increase in the partial dependence value.
# 
# Complex Relationship: The fluctuations between 20 and 40 suggest a more intricate relationship, indicating the need for further analysis to understand the underlying factors.
# 
# Overall Trend: Despite fluctuations, the overall trend indicates that higher "job_desc_complexity" values tend to positively influence the target variable.

# In[ ]:





# In[95]:


#PartialDependence Plot for Feature3
feature_name = "resume_jd_similarity"
feature_index = X_test.columns.get_loc(feature_name)

PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_test, 
    features=[feature_index], 
    feature_names=X_test.columns, 
    grid_resolution=50,  
    kind="average"
)
plt.title(f"Partial Dependence Plot for {feature_name}")
plt.show()


# Higher Sensitivity at Lower Scores:
# 
# The model is more sensitive to changes in "resume_jd_similarity" when the similarity score is low (0.1 to 0.3). Small increases in similarity within this range significantly improve the predicted outcome.
# 
# Diminishing Returns:
# 
# Beyond a similarity score of 0.3, further increases in "resume_jd_similarity" result in smaller improvements in the predicted outcome. This indicates that once a certain level of similarity is achieved, its additional impact becomes less pronounced.
# 
# Common Range:
# 
# The majority of resumes have lower similarity scores (0.1 to 0.3), highlighting a potential area for improvement in aligning resumes with job descriptions to achieve better predictions.
# 
# 
# The concentration of data points at lower similarity scores suggests that most resumes in the dataset have lower similarity scores with job descriptions.
# 
# 
# The stabilization of partial dependence beyond 0.3 indicates that further increases in "resume_jd_similarity" have a less significant impact on the predicted outcome.

# 

# 2D Partial Dependence Plot

# In[96]:


# The top 2 features are cultural_fit_sentiment and clarity_score

features = [('cultural_fit_sentiment', 'clarity_score')]  # 2D feature tuple

print("\n--- 2D Partial Dependence Plot for 'cultural_fit_sentiment' and 'clarity_score' ---")
PartialDependenceDisplay.from_estimator(
    estimator=xgb_best,
    X=X_train,  
    features=features, 
    grid_resolution=50,
    kind='average',
)

# Show the plot
plt.suptitle("2D Partial Dependence Plot", fontsize=16)
plt.tight_layout()
plt.show()


# As "cultural_fit_sentiment" increases, there is a gradual increase in the partial dependence value, which suggests a positive relationship between cultural fit sentiment and the target variable.
# 
# Similarly, as "clarity_score" increases, the partial dependence value also rises, indicating a positive impact of clarity score on the target variable.
# 
# The combined effect of high "cultural_fit_sentiment" and high "clarity_score" results in the highest partial dependence values, suggesting that these two variables together significantly influence the model's predictions.
# 
# When clarity_score is low (less than 55), the partial dependence values are higher, ranging from ~0.46 to ~0.51. This suggests that lower clarity_score leads to higher model predictions, regardless of the cultural_fit_sentiment.
# 
# As clarity_score increases beyond 60, the partial dependence values decline sharply, ranging from ~0.11 to ~0.36. This decline is more pronounced for higher cultural_fit_sentiment values (greater than 4).

# 

# In[ ]:





# In[107]:


import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb


# In[108]:


data = pd.read_csv('new_combined_data.csv')


# Distil Bert Model

# In[109]:


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


# In[110]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# In[111]:


data.columns


# In[112]:


# Custom Dataset for Batch Processing
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Function to get embeddings batch-wise
def generate_embeddings(texts, batch_size=32):
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            encoded_inputs = tokenizer(
                list(batch), return_tensors='pt', truncation=True, padding=True, max_length=512
            )
            encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}
            outputs = model(**encoded_inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


# In[113]:


text_features = ['Transcript','Resume','Reason for decision', 'Job Description', 'polarity']
numerical_features = [
    'num_words_in_transcript', 'resume_jd_similarity', 
    'resume_transcript_similarity', 'sentiment',
    'lexical_diversity', 'transcript_length', 'technical_skill_match',
    'soft_skills_sentiment', 'resume_length',
    'job_description_experience_match', 'cultural_fit_sentiment',
    'job_fit_score', 'confidence_score', 'job_desc_complexity',
    'interaction_quality', 'clarity_score', 
    'text_complexity_transcript', 'text_complexity_resume'
]


# In[ ]:


# Generate embeddings
for feature in text_features:
    print(f"Generating embeddings for {feature}...")
    data[f'{feature}_embedding'] = list(generate_embeddings(data[feature].tolist()))

print("All embeddings generated successfully.")

# Normalize numerical features
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Concatenate all features
embeddings = np.concatenate(
    [np.vstack(data[f'{feat}_embedding'].to_numpy()) for feat in text_features] +
    [data[numerical_features].to_numpy()], axis=1
)


# In[ ]:


# Save concatenated embeddings and features back to the dataset
embedding_df = pd.DataFrame(
    embeddings, 
    columns=[f"feature_{i}" for i in range(embeddings.shape[1])]
)

# Add the target column back for supervised learning
embedding_df['decision'] = data['decision'].values


# In[ ]:


embedding_df


# In[ ]:


# Encode target variable
y = data['decision']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)


# In[ ]:


# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define the objective function
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    # Silent training with verbose=0
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, preds)

    # Print ROC and AUC score for the current trial
    print(f"Trial completed - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    return roc_auc  # Optimize for AUC

# Run the study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Print the best parameters and AUC
print("\nBest parameters:", study.best_params)
print("Best ROC-AUC score:", study.best_value)


# In[ ]:


# Best parameters and final model training
best_params = study.best_params
print("Best Hyperparameters:", best_params)


# In[ ]:


final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
final_model.fit(X_train, y_train)


# In[ ]:


# Predictions and evaluation
y_pred = final_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC_AUC:", roc_auc_score(y_test, y_pred))


# In[ ]:


y_test_pred_classes = final_model.predict_proba(X_test)
y_test_pred_xgb_distil = np.argmax(y_test_pred_classes, axis=1)


# In[ ]:





# In[ ]:





# Sentence Transformer

# In[ ]:


from sentence_transformers import SentenceTransformer

# Define a custom dataset for efficient DataLoader usage
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Function to generate embeddings using CLS Token Pooling with SBERT
def generate_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32, max_length=512):
    """
    Generates embeddings using Sentence Transformers with CLS token pooling.
    Args:
        texts (list): List of texts to embed.
        model_name (str): Pre-trained SentenceTransformer model.
        batch_size (int): Batch size for embedding generation.
        max_length (int): Maximum token length for each text.
    Returns:
        np.ndarray: Generated embeddings.
    """
    # Load the SentenceTransformer model
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_length  # Adjust max token length

    # Dataset and DataLoader for batch processing
    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Tokenize and encode with CLS token pooling
            batch_embeddings = model.encode(
                batch, 
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Ensures cosine similarity compatibility
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


# In[ ]:


text_features = ['Transcript','Resume','Reason for decision', 'Job Description','polarity']
numerical_features = [
    'num_words_in_transcript', 'resume_jd_similarity', 
    'resume_transcript_similarity', 'sentiment',
    'lexical_diversity', 'transcript_length', 'technical_skill_match',
    'soft_skills_sentiment', 'resume_length',
    'job_description_experience_match', 'cultural_fit_sentiment',
    'job_fit_score', 'confidence_score', 'job_desc_complexity',
    'interaction_quality', 'clarity_score', 
    'text_complexity_transcript', 'text_complexity_resume'
]


# In[ ]:


# Generate embeddings
for feature in text_features:
    print(f"Generating embeddings for {feature}...")
    data[f'{feature}_embedding'] = list(generate_embeddings(data[feature].tolist()))

print("All embeddings generated successfully.")

# Normalize numerical features
scaler = MinMaxScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Concatenate all features
embeddings1 = np.concatenate(
    [np.vstack(data[f'{feat}_embedding'].to_numpy()) for feat in text_features] +
    [data[numerical_features].to_numpy()], axis=1
)


# In[ ]:


# Save concatenated embeddings and features back to the dataset
embedding_df1 = pd.DataFrame(
    embeddings1, 
    columns=[f"feature_{i}" for i in range(embeddings1.shape[1])]
)

# Add the target column back for supervised learning
embedding_df1['decision'] = data['decision'].values


# In[ ]:


embedding_df1


# In[ ]:


# Encode target variable
y = data['decision']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings1, y, test_size=0.2, random_state=42)


# In[ ]:


# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define the objective function
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }

    # Silent training with verbose=0
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, preds)

    # Print ROC and AUC score for the current trial
    print(f"Trial completed - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    return roc_auc  # Optimize for AUC

# Run the study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Print the best parameters and AUC
print("\nBest parameters:", study.best_params)
print("Best ROC-AUC score:", study.best_value)


# In[ ]:


best_params = study.best_params
final_model2 = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
final_model2.fit(X_train, y_train)


# In[ ]:


# Predictions and evaluation
y_pred = final_model2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC_AUC:", roc_auc_score(y_test, y_pred))


# In[ ]:


y_test_pred_classes = final_model2.predict_proba(X_test)
y_test_pred_xgb_sen= np.argmax(y_test_pred_classes, axis=1)


# In[ ]:


# Save the model using pickle
with open('sentence_transformer_model.pkl', 'wb') as file:
    pickle.dump(final_model2, file)


# In[ ]:





# In[ ]:





# ANN

# In[ ]:


import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to create and train the model for Optuna optimization
def objective(trial):
    model = Sequential()
    
    # Hyperparameter tuning for number of units and layers
    units_1 = trial.suggest_int('units_1', 32, 256, step=32)
    dropout_1 = trial.suggest_float('dropout_1', 0.1, 0.5, step=0.1)
    units_2 = trial.suggest_int('units_2', 32, 128, step=32)
    dropout_2 = trial.suggest_float('dropout_2', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    # Build the model
    model.add(Dense(units=units_1, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(rate=dropout_1))
    model.add(Dense(units=units_2, activation='relu'))
    model.add(Dropout(rate=dropout_2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=0)
    
    # Evaluate the model on validation data
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]  # Return validation accuracy

# Create an Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Get the best parameters and train the final model
best_params = study.best_params
print("Best Hyperparameters:", best_params)


# In[ ]:


# Final model using the best parameters
final_model1 = Sequential()
final_model1.add(Dense(units=best_params['units_1'], activation='relu', input_dim=X_train.shape[1]))
final_model1.add(Dropout(rate=best_params['dropout_1']))
final_model1.add(Dense(units=best_params['units_2'], activation='relu'))
final_model1.add(Dropout(rate=best_params['dropout_2']))
final_model1.add(Dense(1, activation='sigmoid'))

final_model1.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])

# Train the final model
final_model1.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the final model
final_accuracy = final_model1.evaluate(X_test, y_test)
print(f"Final Model Test Accuracy: {final_accuracy[1]:.4f}")


# In ANN, I got a accuracy of 88.66

# In[ ]:


y_test_pred_nn = final_model1.predict(X_test)


# In[ ]:


test_df = pd.DataFrame()
test_df['actual'] = y_test
test_df['xg_distil_bert'] = y_test_pred_xgb_distil
test_df['xg_sen_transformer'] = y_test_pred_xgb_sen
test_df['nn_prediction'] = y_test_pred_nn


# In[ ]:


test_df


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred),
    }

# Majority voting
test_df['ensemble_vote'] = (test_df[['xg_distil_bert', 'xg_sen_transformer', 'nn_prediction']].mean(axis=1) > 0.5).astype(int)

# Evaluate ensemble performance
ensemble_metrics = calculate_metrics(test_df['actual'], test_df['ensemble_vote'])
print("Ensemble Metrics:", ensemble_metrics)


# In[ ]:


# Mean

test_df['mean_prob'] = (test_df['xg_distil_bert'] + test_df['xg_sen_transformer'] + test_df['nn_prediction'])/3


# In[ ]:


test_df['new_pred'] = test_df['mean_prob'].round()
test_df['new_nn_prediction'] = test_df['nn_prediction'].round()
test_df


# Accuracy for actual with ensembing

# In[ ]:


print(accuracy_score(test_df['actual'], test_df['new_pred']))
print(roc_auc_score(test_df['actual'], test_df['new_pred']))


# Accuracy for actual with distil bert

# In[ ]:


print(accuracy_score(test_df['actual'], test_df['xg_distil_bert']))
print(roc_auc_score(test_df['actual'], test_df['xg_distil_bert']))


# Accuracy for actual with sentence transformer

# In[ ]:


print(accuracy_score(test_df['actual'], test_df['xg_sen_transformer']))
print(roc_auc_score(test_df['actual'], test_df['xg_sen_transformer']))


# Accuracy for actual with ANN

# In[ ]:


print(accuracy_score(test_df['actual'], test_df['new_nn_prediction']))
print(roc_auc_score(test_df['actual'], test_df['new_nn_prediction']))


# In[ ]:


print(roc_auc_score(test_df['actual'], test_df['nn_prediction']))


# In[ ]:




