from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import TweetTokenizer
import re, string
import csv
import random

tknzr = TweetTokenizer()

cnt=0

time_sensitive_tweet_tokens=[]

non_sensitive_tweet_tokens=[]

#tokenization, lemmatization, noise removal

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
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

#reading in tweets and organizing them

with open(r'C:\Users\34346\Desktop\Code\Research\7-24 to 7-31\TE-organized-tweets-sample.csv','r',encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if cnt == 0:
            cnt+=1
        else:
            cnt+=1
            temp_tokens=tknzr.tokenize(row[2])
            #print(temp_tokens)
            if (row[11]=='Y'):
                time_sensitive_tweet_tokens.append(temp_tokens)
            else:
                non_sensitive_tweet_tokens.append(temp_tokens)

#print(time_sensitive_tweet_tokens)

time_sensitive_cleaned_tokens_list=[]
non_sensitive_cleaned_tokens_list=[]

for tokens in time_sensitive_tweet_tokens:
    time_sensitive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in non_sensitive_tweet_tokens:
    non_sensitive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
#print(time_sensitive_cleaned_tokens_list)

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(time_sensitive_cleaned_tokens_list)

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10))

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

time_sensitive_tokens_for_model = get_tweets_for_model(time_sensitive_cleaned_tokens_list)
non_sensitive_tokens_for_model = get_tweets_for_model(non_sensitive_cleaned_tokens_list)

time_sensitive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in time_sensitive_tokens_for_model]

non_sensitive_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in non_sensitive_tokens_for_model]

dataset = time_sensitive_dataset + non_sensitive_dataset

random.shuffle(dataset)

maxn=65
train_data = dataset[:maxn]
test_data = dataset[maxn:]

from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

from nltk.tokenize import word_tokenize

custom_tweet = input()

custom_tokens = remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token, True] for token in custom_tokens)))
