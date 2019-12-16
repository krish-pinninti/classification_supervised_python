import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv("creditcard.csv")

dataset.head(10)

dataset.tail()

dataset.describe()

dataset.info()

normal = dataset[dataset['Class']==0]
fraud = dataset[dataset['Class']==1]

fraud.shape
normal.shape

sns.countplot(dataset['Class'], label = "Count") 

plt.figure(figsize
sns.heatmap(dataset.corr(), annot=True) =(30,10)) 
# Most of the dataset is uncorrelated, its probably because the data is a result of Principal Componenet Analysis (PCA)
# Features V1 to V28 are Principal Components resulted after propagating real features through PCA. 

column_headers = dataset.columns.values

# kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable.
i = 1

fig, ax = plt.subplots(8,4,figsize=(18,30))
for column_header in column_headers:    
    plt.subplot(8,4,i)
    sns.kdeplot(fraud[column_header], bw = 0.4, label = "Fraud", shade=True, color="r", linestyle="--")
    sns.kdeplot(non_fraud[column_header], bw = 0.4, label = "Normal", shade=True, color= "y", linestyle=":")
    plt.title(column_header, fontsize=12)
    i = i + 1
plt.show();



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset['Amount_Norm'] = sc.fit_transform(dataset['Amount'].values.reshape(-1,1))

dataset = dataset.drop(['Amount'], axis = 1)


# Let's drop the target label coloumns
X = dataset.drop(['Class'],axis=1)
y = dataset['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.shape

y_train.shape

X_test.shape
y_test.shape

from sklearn.naive_bayes import GaussianNB 
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
cm
sns.heatmap(cm, annot=True)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print("Accuracy Score :")
print(accuracy_score(y_test, y_predict_test))
print("Classification Report :")
print(classification_report(y_test, y_predict_test))

#improving the model
X = dataset.drop(['Time','V8','V13','V15','V20','V22','V23','V24','V25','V26','V27','V28','Class'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_predict = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
cm
sns.heatmap(cm, annot=True)

print("Accuracy Score :")
print(accuracy_score(y_test, y_predict_test))
print("Classification Report :")
print(classification_report(y_test, y_predict_test))

print("Number of fraud points in the testing dataset = ", sum(y_test))