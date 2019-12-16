import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


dataset = pd.read_csv("emails.csv")

dataset.head(10)

dataset.tail()

dataset.describe()
dataset.info()

spam = dataset[dataset['spam']==1]
nonspam = dataset[dataset['spam']==0]

spam.shape
nonspam.shape

sns.countplot(dataset['spam'], label = "Count") 


#apply countVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
textdata_countvectorizer = vectorizer.fit_transform(dataset['text'])

print(vectorizer.get_feature_names())

print(textdata_countvectorizer.toarray())  

textdata_countvectorizer.shape


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = dataset['spam'].values
NB_classifier.fit(textdata_countvectorizer, label)

testing_sample = ['Free money!!!', "Hi Kim, Please let me know if you need any further information. Thanks"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)


test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


testing_sample = ['money viagara!!!!!', "Hello, I am Selva, I would like to book a hotel in SF by January 24th"]
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


#data into training and testing

X = textdata_countvectorizer
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# from sklearn.naive_bayes import GaussianNB 
# NB_classifier = GaussianNB()
# NB_classifier.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


print(classification_report(y_test, y_predict_test))

