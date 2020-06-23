#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from flair.embeddings import Sentence
from flair.embeddings import FlairEmbeddings


# In[2]:


inputFileName = "../corpusTemporal/corpusTemporal.p"
outputFileName = "vectors.txt"


# In[3]:


char_lm_embeddings = FlairEmbeddings('resources/taggers/language_model/best-lm.pt')


# In[4]:


corpus = pickle.load(open(inputFileName, 'br'))


# In[5]:


vectors = {}


# In[6]:


for d in corpus:
    print("processing ",d)
    totLen = len(corpus[d]['text'])
    for i,s in enumerate(corpus[d]['text']):
        if i%10 == 0:
            print("processed {}/{}        ".format(i,totLen), end='\r')
        sentence = Sentence(s)
        char_lm_embeddings.embed(sentence)
        for token in sentence:
            if not token.text in vectors:
                string = token.text
                for v in token.embedding.cpu().numpy():
                    string += ' {}'.format(v)
                vectors[token.text] = string
    print("processed {}/{}        ".format(i,totLen))


# In[11]:


with open(outputFileName, 'wt') as f:
    for k in vectors:
        f.write(vectors[k])
        f.write("\n")

