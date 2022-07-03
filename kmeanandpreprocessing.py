#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
folderpath = r"C:\Users\user\Downloads\Compressed\wiki"
filepaths = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files = []
for path in filepaths:
    with open(path, 'r', encoding = "ISO-8859-1") as f:
        file = f.readlines()
        all_files.append(file)


# In[ ]:


all_files


# In[ ]:


sizeOfmyfiles = len(all_files)
print(sizeOfmyfiles)


# In[ ]:


listToStr =' '.join(map(str, all_files))


# In[ ]:


listToStr


# In[ ]:


import nltk
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[ ]:





# In[ ]:


from nltk.tokenize import word_tokenize


# In[ ]:


listToStr


# In[ ]:


from nltk.tokenize import word_tokenize
text = word_tokenize(listToStr)


# In[ ]:


data = [word for word in text if not word in stopwords.words()]
print(data)


# In[ ]:


nltk.download('wordnet')


# In[ ]:


from nltk.stem import WordNetLemmatizer


# In[ ]:


lemma = WordNetLemmatizer()


# In[ ]:


lem = []


# In[ ]:


nltk.download('omw-1.4')


# In[ ]:


for r in data:
    lem.append(lemma.lemmatize(r))


# In[ ]:


print(lemma.lemmatize('town'))


# In[ ]:


import re
dataupdate = []
dataupdate = [re.sub('[^a-zA-Z0-9]', '', _) for _ in lem]
dataupdate


# In[ ]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#import warnings filter
from warnings import simplefilter
#ignore all future warnings
simplefilter(action = 'ignore', category = FutureWarning)


vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(dataupdate)

true_k = 5
model = KMeans(n_clusters = true_k, init = 'k-means++', max_iter = 100, n_init = 1)
model.fit(X)

print("Top terms per cluster: ")
order_centroids = model.cluster_centers_.argsort()[:, :: -1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    print("Cluster %d : " % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),


# In[ ]:


print("Guessing : ")

Y = vectorizer.transform(["age"])
guessing = model.predict(Y)
print(guessing)

Y = vectorizer.transform(["people"])
guessing = model.predict(Y)
print(guessing)


# In[ ]:




