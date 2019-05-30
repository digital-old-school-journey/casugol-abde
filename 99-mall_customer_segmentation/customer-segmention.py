#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '99-mall_customer_segmentation'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Customer Segmentation Project
# ### In this project, I will try to figure out who the best customers are. I will look at the data in the following ways to answer that question:
# #### 1. Explore the general distribution of the data to get a sense of Male vs. Female customers, and how their income, age, and spending scores are similar or different.
# #### 2. Explore which gender has a higher income.
# #### 3. Explore which gender has a higher average spending score.
#%% [markdown]
# ## Preparation

#%%
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%%
customers = pd.read_csv('Mall_Customers.csv')


#%%
customers.head()

#%% [markdown]
# ## EDA

#%%
# Check to see descriptive statistics
customers.describe()

#%% [markdown]
# #### From calling describe, we can see that there are no values to clean. Age looks pretty normally distributed, annual income in the thousands doesn't have outliers that are too excessive. Spending score is in fact between 1 and 100. Everything looks good.

#%%
# See the distribution of gender to recognize different distributions
sns.countplot(x='Gender', data=customers);
plt.title('Distribution of Gender');

#%% [markdown]
# There are more women than men in this dataset.

#%%
# Histogram of ages
customers.hist('Age', bins=35);
plt.title('Distribution of Age');
plt.xlabel('Age');

#%% [markdown]
# #### Age histogram is somewhat right-tailed. We saw that the average age was 38 as well, so this is not surprising, with a spike in ages 48-49 and 65 as well.

#%%
# Histogram of ages by gender
plt.hist('Age', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Male');
plt.hist('Age', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Female');
plt.title('Distribution of Age by Gender');
plt.xlabel('Age');
plt.legend();

#%% [markdown]
# ### We can see two things here, one reflected earlier in the describe call:
# 
# #### 1. There are more women than men in this data set.
# 
# #### 2. There are a lot of younger women and middle-aged women.

#%%
# Histogram of income
customers.hist('Annual Income (k$)');
plt.title('Annual Income Distribution in Thousands of Dollars');
plt.xlabel('Thousands of Dollars');

#%% [markdown]
# #### The most frequent annual incomes are between around 50 and 85,000.

#%%
# Histogram of income by gender
plt.hist('Annual Income (k$)', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Male');
plt.hist('Annual Income (k$)', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Female');
plt.title('Distribution of Income by Gender');
plt.xlabel('Income (Thousands of Dollars)');
plt.legend();

#%% [markdown]
# #### Women generally had lower income than men, the majority falling between 45 and 80,000.

#%%
# Create data sets by gender
male_customers = customers[customers['Gender'] == 'Male']
female_customers = customers[customers['Gender'] == 'Female']


#%%
# Print the average spending score for men and women
print(male_customers['Spending Score (1-100)'].mean())
print(female_customers['Spending Score (1-100)'].mean())

#%% [markdown]
# #### Women on average had a higher spending score by about 3 points.

#%%
sns.scatterplot('Age', 'Annual Income (k$)', hue='Gender', data=customers);
plt.title('Age to Income, Colored by Gender');

#%% [markdown]
# #### There is pretty much no correlation between age and income for either men or women in this data. The correlation matrix below confirms this.
sns.heatmap(customers.corr(), annot=True);
plt.title('Heatmap Correlation of All Variables');
#%%
sns.scatterplot('Age', 'Spending Score (1-100)', hue='Gender', data=customers);
plt.title('Age to Spending Score, Colored by Gender');

#%% [markdown]
# #### The above plot shows the negative correlation between age and spending score. It's not a strong association, but the older the person, the worse their spending score.

#%%
sns.heatmap(female_customers.corr(), annot=True);
plt.title('Correlation Heatmap - Female');


#%%
sns.heatmap(male_customers.corr(), annot=True);
plt.title('Correlation Heatmap - Male');

#%% [markdown]
# #### Comparing men to women reveals that there is a slightly higher correlation between age and spending score for women.

#%%
sns.lmplot('Age', 'Spending Score (1-100)', data=female_customers);
plt.title('Age to Spending Score, Female Only');


#%%
sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)', hue='Gender', data=customers);
plt.title('Annual Income to Spending Score, Colored by Gender');

#%% [markdown]
# #### Annual Income and Spending Score have somewhat related means and standard deviations, which is why we see this strange shape. But there is very low if no correlation here.
#   
# #### There are 5 rough clusters here:
# 1. Low income, low spending score
# 2. Low income, high spending score
# 3. Mid income, medium spending score
# 4. High income, low spending score
# 5. High income, high spending score
# 
# Interestingly, there are no high income, medium spending score points.
#%% [markdown]
# ## Interpretation and Actions
#%% [markdown]
# ### Based on these data, the following hypotheses could be tested:
# 
# 1. Marketing cheaper items to women to see if they purchase more frequently or more volume.
# 
# 2. Marketing more to younger women because their spending score tends to be higher.
# 
# 3. Thinking up new ways to target advertising, pricing, branding, etc. to the older women (older than early 40s) who have lower spending scores.
# 
# 4. Figure out a way to gather more data to build a data set that has more features. The more features, the better understanding of what determines Spending Score. Once Spending Score is better understood, we can understand what factors will lead to increasing Spending Score, thus lead to greater profits.
# 
# ### KPIs
# 
# In the spirit of business use cases, I'll define the following KPIs as an example to show how you would know if your efforts are paying off or not. 
# 1. The change in frequency and volume of purchases by women after the introduction of more marketing campaigns targeting them.
# 2. The change in spending score after introducing marketing campaigns targeting younger women.
# 3. The change in spending score after introducing marketing campaigns targeting older women.

