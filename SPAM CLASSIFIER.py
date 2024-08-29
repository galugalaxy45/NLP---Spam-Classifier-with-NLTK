#!/usr/bin/env python
# coding: utf-8

# # <span style = "color:green"> Spam Classifier with NLTK </spam>

# ***

# The SMS spam collection dataset is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5574 messages, tagged according being ham (legitimate) or spam.

# ### Content

# The file contain one message per line. Each line is composed by two columns:
# v1 contains the label (ham or spam) and v2 contains the raw text.

# ## Let's Begin

# ### Import necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns


# ### Read 'spam.csv' dataset

# In[2]:


df = pd.read_csv('spam.csv', encoding = 'ISO-8859-1')


# ### Check the head

# In[3]:


df.head()


# ### Drop all Unnamed columns

# In[4]:


df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)


# In[5]:


df.head()


# ### Check info 

# In[6]:


df.info()


# ### Rename v1 to Label and v2 to Messages

# In[7]:


df.rename(columns = {'v1':'Label','v2':'Messages'}, inplace = True)


# In[8]:


df.head()


# ### Print few of the messages

# In[9]:


for i in range(10):
    print(df['Messages'][i])


# ### Check the unique values in label

# In[10]:


df['Label'].unique()


# ### Check for null values

# In[11]:


df.isna().sum()


# ### Check for duplicates

# In[12]:


df.duplicated().sum()


# Looks like we have 403 duplicate values

# ### Drop duplicates

# In[13]:


df.drop_duplicates(keep='first', inplace = True)


# In[14]:


df.head()


# ### Confirm droped duplicates

# In[15]:


df.duplicated().sum()


# ## Exploratory Data Analysis

# ### Check the value counts in the dataset

# In[16]:


df['Label'].value_counts()


# ### Visualize the value count using pieplot

# In[17]:


plt.pie(df['Label'].value_counts(), labels = ['ham','spam'], autopct='%0.2f')
plt.show()


# The dataset seem to be slightly imbalanced

# ### Import WordNetLemmatizer, stopwords

# In[18]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# ### Create a function to clean the messages

# In[19]:


def preprocess(sentence):
    #removes all the special characters and split the sentence at spaces
    text = re.sub(r"[^a-zA-Z0-9]"," ",sentence).split()
    
    # converts words to lowercase and removes any stopwords
    words = [x.lower() for x in text if x not in stopwords.words('english')]
    
    # Lemmatize the words
    lemma = WordNetLemmatizer()
    word = [lemma.lemmatize(word,'v') for word in words ]
    
    # convert the list of words back into a sentence
    word = ' '.join(word)
    return word


# ### Apply the function to messages feature in our dataframe

# In[20]:


df['Messages'] = df['Messages'].apply(preprocess)


# In[21]:


df.head()


# ### Print few of the sentences after preprocessing

# In[22]:


for i in range(10):
    print(df['Messages'][i])


# ### Create Bag of Words model

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer


# In[24]:


cv = CountVectorizer()


# ### Fit transform messages feature in our dataframe

# In[25]:


X = cv.fit_transform(df['Messages']).toarray()


# ### Check the shape of X

# In[26]:


X.shape


# ### Print X

# In[27]:


print(X)


# ### Create a function to change 'ham' to 1 and 'spam' to 0 in our label features

# In[28]:


def hamspam(x):
    if x == 'ham':
        return 1
    else:
        return 0


# ### Apply the function to our label features in our dataframe and store it in y variable

# In[29]:


y = df['Label'].apply(hamspam)


# In[30]:


y


# ### Split the dataset into training and testing set

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# ### Create Random Forest model

# In[33]:


from sklearn.ensemble import RandomForestClassifier


# In[34]:


model = RandomForestClassifier()


# ### Train the model

# In[35]:


model.fit(X_train, y_train)


# ### Check the score of our training set

# In[36]:


model.score(X_train, y_train)


# ### Make prediction with X_test

# In[37]:


prediction = model.predict(X_test)


# ### Check the accuracy of our prediction

# In[38]:


from sklearn import metrics


# In[39]:


metrics.accuracy_score(y_test, prediction)


# ### Visualize confusion matrix on a heatmap

# In[40]:


sns.heatmap(metrics.confusion_matrix(y_test,prediction),annot = True, fmt = 'd')
plt.show()


# ### Create classification report

# In[41]:


print(metrics.classification_report(y_test,prediction))

