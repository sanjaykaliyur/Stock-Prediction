
# coding: utf-8

# Install Tweepy. Allows access to Twitter API.

# In[1]:

get_ipython().system(u'pip install tweepy')


# Install Textblob. Used for Natural Language Processing. Goes hand-in-hand with Tweepy for Sentiment Analysis.

# In[2]:

get_ipython().system(u'pip install -U textblob')
get_ipython().system(u'python -m textblob.download_corpora')


# Install Requests. HTTP library for Python.

# In[3]:

get_ipython().system(u'pip install requests')


# Install Keras. Built on TensorFlow.

# In[4]:

get_ipython().system(u'pip install keras')


# Import required libraries. 

# In[5]:

import os
import tweepy
import requests
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from textblob import TextBlob


# Login to Twitter. Requires you to enter consumer_key, consumer_secret, access_token, and access_token_secret. All can be accessed through [dev.twitter.com](https://dev.twitter.com).

# In[6]:

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)


# Create a temporary CSV file to hold our data.

# In[7]:

# Where the csv file will live
data = 'data.csv'


# Sentiment Analysis function

# In[8]:

def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
    # majority of valid tweets have positive sentiment
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1

    if positive > ((num_tweets - null)/2):
        return True


# In[9]:

def get_data(stock):
    # Download our file from google finance
    url = 'http://www.google.com/finance/historical?q=NASDAQ%3A'+stock+'&output=csv'
    r = requests.get(url, stream=True)

    if r.status_code != 400:
        with open(data, 'wb') as f:
            for chunk in r:
                f.write(chunk)
        return True


# In[22]:

def stock_prediction(stock):

    # Collect data points from csv
    dataset = []

    with open(data) as f:
        for n, line in enumerate(f):
            if n != 0:
                dataset.append(float(line.split(',')[1]))

    dataset = np.array(dataset)

    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        return np.array(dataX), dataset[2:]
        
    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=0)

    # Our prediction for tomorrow
    prediction = model.predict(np.array([dataset[0]]))
    #result = '%s stock price will move from %s to %s' % (stock, dataset[0], prediction[0][0])
    result = "%s's stock price will move from %s to %s on the next open day of the stock exchange." % (stock, dataset[0], prediction[0][0])
    return result


# In[23]:

# Ask user for a stock quote
stock = raw_input('Enter a stock quote from NASDAQ (e.g: AAPL, FB, GOOGL): ').upper()


# In[24]:

num_tweets = int(input('How many tweets should I look through to determine the sentiment about %s? ' % (stock)))


# In[25]:

get_data(stock)


# Sentiment check of the stock being analyzed.

# In[26]:

if stock_sentiment(stock, num_tweets):
    print 'This stock has good sentiment on Twitter.'
    
if not stock_sentiment(stock, num_tweets):
    print 'This stock has bad sentiment on Twitter.'


# In[27]:

# We have our file so we create the neural net and get the prediction
print stock_prediction(stock)

# We are done so we delete the csv file
os.remove(data)


# In[ ]:



