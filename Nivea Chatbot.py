#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import nltk
import io #opening and reading a file 
import numpy as np # doing some numerical calculations
import random # selecting the random words from text
import string # to process a standard python strings
import warnings # while running our application code if any warning comes it will be ignored by using below comment
warnings.filterwarnings('ignore')


# In[17]:


data=open('Beiersdorf Overview.txt','r', errors='ignore')
raw=data.read()
raw=raw.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens= nltk.sent_tokenize(raw)
word_tokens=nltk.word_tokenize(raw)


# In[21]:


sent_tokens[:2]
word_tokens[:2]


# In[23]:


lemmer=nltk.stem.WordNetLemmatizer()
#WordNet is a semmantically-oriented dictionary of English included in NLTK
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[24]:


GREETING_INPUT =("hello","hi","greetings","sup","what's up","hey")
GREETING_RESPONSE=["hi","hey","*nods*","hi there","hello","I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUT:
            return random.choice(GREETING_RESPONSE)


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# In[28]:


def response(user_response):
    chatbot_response=" "
    sent_tokens.append(user_response)
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        chatbot_response=chatbot_response+"I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response= chatbot_response+ sent_tokens[idx]
        return chatbot_response


# In[ ]:


flag=True
print("Chatbot: My name is Chatbot. I will answer your queries about Beiersdorf. If you want to exit, type bye!")
while(flag==True):
    user_response=input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("Chatbot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Chatbot: "+greeting(user_response))
            else:
                print("chatbot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Chatbot: Bye! take care..")


# In[ ]:




