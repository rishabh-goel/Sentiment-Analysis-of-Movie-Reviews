# your code goes here
import nltk
import os
 
l = os.listdir('/home/rishabh/review_polarity/txt_sentoken/pos')
pos = []
for file in l:
    content = open(os.path.join('/home/rishabh/review_polarity/txt_sentoken/pos', file), 'r')
    buf = content.read()
    tokens = nltk.word_tokenize(buf)
    tagged_tokens = nltk.pos_tag(tokens)
    rel_tokens = ""
    for word in tagged_tokens:
        if word[1] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS']: 
            rel_tokens += word[0] + ' '
    rel_tokens = rel_tokens[:-1]
    rel_tokens.replace('not ', 'not-')
    pos.append(rel_tokens)
 
l = os.listdir('/home/rishabh/review_polarity/txt_sentoken/neg')
neg = []
for file in l:
    content = open(os.path.join('/home/rishabh/review_polarity/txt_sentoken/neg', file), 'r')
    buf = content.read()
    tokens = nltk.word_tokenize(buf)
    tagged_tokens = nltk.pos_tag(tokens)
    rel_tokens = ""
    for word in tagged_tokens:
        if word[1] in ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS']:  
            rel_tokens += word[0] + ' '
    rel_tokens = rel_tokens[:-1]
    rel_tokens.replace('not ', 'not-')
    neg.append(rel_tokens)
 
result = [None]*(len(pos)+len(neg))
result[::2] = pos
result[1::2] = neg
 
 
import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
 
sample_data_training = result[:int(len(result) * 0.75)]
labels_training = [1, 0] * (int((len(result) * 0.75)/2))
sample_data_testing = result[int(len(result) * 0.75):]
labels_testing = [1, 0] * (int((len(result) * 0.25)/2))
model_list = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
for model in model_list:
    pipe = Pipeline([('vectorizer', CountVectorizer()), ('classifier', model)])
    pipe.fit(sample_data_training, labels_training)
    #print(pipe.named_steps['vectorizer'].vocabulary_)
    predictions = pipe.predict(sample_data_testing)
    for prediction in predictions:
        polarity = 'good' if prediction == 1 else 'bad'
        #print('This movie is ' + polarity)  # Can we somehow print the name of the movie here?
    print(accuracy_score(labels_testing, predictions))
    print(classification_report(labels_testing, predictions))
    print(confusion_matrix(labels_testing, predictions))