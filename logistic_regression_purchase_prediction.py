# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset.shape

dataset.describe()
dataset.isnull().values.any()

X = dataset.iloc[:, [2, 3]].values
y = dataset["Purchased"].values

dataset.isnull().values.any()

not_purchased    = dataset[dataset['Purchased']==0]
purchased = dataset[dataset['Purchased']>0]

dataset.shape
purchased.shape
not_purchased.shape

def showBarChart(data_x, data_y):
    plt.figure(figsize=[6,12])
    plt.subplot(211)
    sns.countplot(x = data_x, data = dataset)
    plt.subplot(212)
    sns.countplot(x = data_x, hue = data_y, data=dataset)

showBarChart('Age', 'Purchased')
showBarChart('EstimatedSalary', 'Purchased')


y.shape
X.shape
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm
sns.heatmap(cm, annot=True, fmt="d")


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
def displayVisualMap(X_set, y_set, setType):
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Logistic Regression ( set)' + setType)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
    
displayVisualMap(X_train, y_train, 'Training')
displayVisualMap(X_test, y_test, 'Test')

