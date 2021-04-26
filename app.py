from flask import Flask, request, render_template, jsonify
import os
import pickle
from nltk import data
import pandas as pd


app = Flask(__name__,   static_folder='static',
            template_folder='templates')

model=pickle.load(open('sentimental_101.pkl', 'rb'))


@app.route('/', methods=['GET','POST'])
def home():
       image_names= os.listdir('./static/images')
       csseff= os.listdir('./static/styles')

       return render_template('index.html',image_names=image_names, csseff=csseff)


@app.route('/predict', methods=['POST'])
def predict():
      
   import nltk
   nltk.download('twitter_samples')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('punkt')

   from nltk.stem.wordnet import WordNetLemmatizer
   from nltk.corpus import twitter_samples, stopwords
   from nltk.tag import pos_tag
   from nltk.tokenize import word_tokenize
   from nltk import FreqDist, classify, NaiveBayesClassifier
   import re, string, random
   import pickle


   def remove_noise(tweet_tokens, stop_words = ()):

      cleaned_tokens = []

      for token, tag in pos_tag(tweet_tokens):
         token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
         token = re.sub("(@[A-Za-z0-9_]+)","", token)

         if tag.startswith("NN"):
               pos = 'n'
         elif tag.startswith('VB'):
               pos = 'v'
         else:
               pos = 'a'

         lemmatizer = WordNetLemmatizer()
         token = lemmatizer.lemmatize(token, pos)

         if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
               cleaned_tokens.append(token.lower())
      return cleaned_tokens

   def get_all_words(cleaned_tokens_list):
      for tokens in cleaned_tokens_list:
         for token in tokens:
               yield token

   def get_tweets_for_model(cleaned_tokens_list):
      for tweet_tokens in cleaned_tokens_list:
         yield dict([token, True] for token in tweet_tokens)

   if __name__ == "__main__":

      positive_tweets = twitter_samples.strings('positive_tweets.json')
      negative_tweets = twitter_samples.strings('negative_tweets.json')
      text = twitter_samples.strings('tweets.20150430-223406.json')
      tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

      stop_words = stopwords.words('english')

      positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
      negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

      positive_cleaned_tokens_list = []
      negative_cleaned_tokens_list = []

      for tokens in positive_tweet_tokens:
         positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

      for tokens in negative_tweet_tokens:
         negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

      all_pos_words = get_all_words(positive_cleaned_tokens_list)

      freq_dist_pos = FreqDist(all_pos_words)
      print(freq_dist_pos.most_common(10))

      positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
      negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

      positive_dataset = [(tweet_dict, "Positive")
                           for tweet_dict in positive_tokens_for_model]

      negative_dataset = [(tweet_dict, "Negative")
                           for tweet_dict in negative_tokens_for_model]

      dataset = positive_dataset + negative_dataset

      random.shuffle(dataset)

      train_data = dataset[:7000]
      test_data = dataset[7000:]

      classifier = NaiveBayesClassifier.train(train_data)

      print("Accuracy is:", classify.accuracy(classifier, test_data))

      print(classifier.show_most_informative_features(10))

      custom_tweet = ""


      if request.method == 'POST':
       	      custom_tweet = request.form['text']

      custom_tokens = remove_noise(word_tokenize(custom_tweet))

      NB_Cls= classifier.classify(dict([token, True] for token in custom_tokens))

      print(custom_tweet, NB_Cls)


      pickle.dump(NB_Cls, open('sentimental_101.pkl', 'wb'))

            
      return render_template('results.html', result= NB_Cls)


if __name__ == "__main__":
   app.run(debug=True,port=1998,host='0.0.0.0')

