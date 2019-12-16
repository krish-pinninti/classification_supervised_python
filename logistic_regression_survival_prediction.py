import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# read the data using pandas dataframe
dataset = pd.read_csv('titanic.csv')

print(dataset.columns)

dataset.shape

# Show the data head!
dataset.head()


# Show the data head!
dataset.tail()

# Let's count the number of survivors and non-survivors

survived    = dataset[dataset['Survived']==1]
no_survived = dataset[dataset['Survived']==0]

survived.shape
no_survived.shape

dataset.describe()

dataset.isnull().values.any()

def showBarChart(data_x, data_y):
    plt.figure(figsize=[6,12])
    plt.subplot(211)
    sns.countplot(x = data_x, data = dataset)
    plt.subplot(212)
    sns.countplot(x = data_x, hue = data_y, data=dataset)


# Bar Chart Class to Survived   
showBarChart('Pclass', 'Survived')

# Bar Chart Siblings to Survived
showBarChart('SibSp', 'Survived')

# Bar Chart Parch to Survived
showBarChart('Parch', 'Survived')

# Bar Chart Embarked to Survived
showBarChart('Embarked', 'Survived')

# Bar Chart Sex to survived
showBarChart('Sex', 'Survived')


# Bar Chart to indicate the number of people survived based on their age
plt.figure(figsize=(40,30))
sns.countplot(x = 'Age', hue = 'Survived', data=dataset)

# Age Histogram 
dataset['Age'].hist(bins = 40)

# Bar Chart to indicate the number of people survived based on their fare
plt.figure(figsize=(40,20))
sns.countplot(x = 'Fare', hue = 'Survived', data=dataset)

# Fare Histogram 
dataset['Fare'].hist(bins = 40)

# Let's explore which dataset is missing
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap="Blues")

# Let's drop the missing data
dataset.drop('Cabin',axis=1,inplace=False)
dataset.drop('Cabin',axis=1,inplace=True)
dataset.drop(['Name', 'Ticket', 'Embarked', 'PassengerId'],axis=1,inplace=True)

dataset.describe()

# Let's view the data one more time!
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# Let's get the average age for male (~29) and female (~25)
plt.figure(figsize=(15, 10))
sns.boxplot(x='Sex', y='Age',data=dataset)

def Fill_Age(data):
    age = data[0]
    sex = data[1]

    if pd.isnull(age):
        if sex is 'male': 
            return 29
        else:
            return 25
    else:
        return age
    
dataset['Age'] = dataset[['Age','Sex']].apply(Fill_Age,axis=1)

# Let's view the data one more time!
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap="Blues")

pd.get_dummies(dataset['Sex'])
# You just need one column only to represent male or female


male = pd.get_dummies(dataset['Sex'], drop_first = True)

# first let's drop the embarked and sex 
dataset.drop(['Sex'], axis=1, inplace=True)

# Now let's add the encoded column male again
dataset = pd.concat([dataset, male], axis=1)


#Let's drop the target coloumn before we do train test split
X = dataset.drop('Survived',axis=1).values
y = dataset['Survived'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

y_predict_test = classifier.predict(X_test)
y_predict_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict_test)
cm
sns.heatmap(cm, annot=True, fmt="d")


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_test))


