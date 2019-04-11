#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#text classification using naive bayes algorithm
#import the data
from sklearn.datasets import fetch_20newsgroups
categories =['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
#get the train folder
news_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
news_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)
#after loading the data,the data in train and the test folder in the form of dictionary
#dictionary mean data in the key value form
#what is dictionary=?
simple_dict = {'key1' :'value1',
               'key2':['lists','of','text']
#                'key3': {'sub_key1':'nested', 'sub_key2':'dictionary'}#also the nested dictionary

}
#how access the element from that dictionary
print(simple_dict.keys())
simple_dict.values()


# In[ ]:


#same like above our test and train also in the form of dictionary
news_train.keys()


# In[ ]:


news_train.target_names


# In[ ]:


#now the countervectorizer step
#it contain the two methods
#one is fit method? this method is used to assign the unique value to each word in the document
#second is transform method?this method is used to calculate the occurance of unique value in the document
#here is the example
text = ["the quick brown fox jumped over the lazy dog",
        "the dog",
        "the fox"]
from sklearn.feature_extraction.text import CountVectorizer
vector =CountVectorizer()
#THE fit method
vector.fit(text)
#basically it return us the vocabulary from the raw document with the unique value of each word
print('vocabulary:' + str(vector.vocabulary_)+'\n\n')
#print he features
print('features are : '+ str(vector.get_feature_names())+'\n\n')
#and now the transform method
count = vector.transform(text)
#print the shape 
print('shape is' + str(count.shape)+'\n\n')#in shape first number repreent the samples , and second represent the features
#print into array
print('array after transform is:'+'\n'+ str(count.toarray()))


# In[ ]:


#now apply the counvectorizer ro our data
coun_vector = CountVectorizer()
train_data_count = coun_vector.fit_transform(news_train.data)#we can also apply fit and transform in one shot
train_data.shape


# In[ ]:


#lets discuss the problem of countervectorizer,actully we know it assign the unique value to each word and also make its
#total of that unique word appear,but we know the word "the",or anyother stop words appear many times,
# and it so many time appearance does not make sense in the document classification
#to solve this problem we use the term frequency or inverse down frequency
#what is term frequency?it summarize how often a given word appear in the document
#what is inverse down frequency? it assign weights to the word according to how much it is important for classification or
#it downscale a word that appear a lots in the documents
#it take the input of the previous counvectorizer array,and perform the fit and transform method to downscale it appearnce
#it is done by the tfidcounvectorizer
#lets start in our ddta
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = TfidfTransformer()
train_data_tdif = vectorizer.fit_transform(train_data_count)
train_data_tdif.shape


# In[ ]:


#now use the multinomial model for training and prediction
from sklearn.naive_bayes import MultinomialNB
mb  =MultinomialNB()
#fit our model on transform data using tfidvectorizer and traget variable
mb.fit(train_data_tdif,news_train.target)


# In[ ]:


#now the preiction step,we give our news test data and transform it,bcz we fit it already,lets see
tets_data_count = coun_vector.transform(news_test.data)
tets_data_tdif = vectorizer.transform(tets_data_count)
#now the prediction
predict = mb.predict(tets_data_tdif)


# In[ ]:


#now the result step
from sklearn import metrics
from sklearn.metrics import accuracy_score
print("model accuracy is " ,accuracy_score(news_test.target,predict)*100)


# In[ ]:




