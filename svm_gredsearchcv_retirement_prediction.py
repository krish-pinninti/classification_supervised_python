# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization

dataset = pd.read_csv('Bank_Customer_retirement.csv')

dataset.keys()

dataset.shape

sns.pairplot(dataset, hue = 'Retire', vars = ['Age', '401K Savings'] )

sns.countplot(dataset['Retire'], label = "Retirement") 

dataset = dataset.drop(['Customer ID'],axis=1)
X = dataset.drop(['Retire'],axis=1)

y = dataset['Retire']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

def executeSVCModel(X_train_inp, X_test_inp):
    svc_model = SVC()
    svc_model.fit(X_train_inp, y_train)    
    y_predict = svc_model.predict(X_test_inp)
    cm = confusion_matrix(y_test, y_predict)
    
    sns.heatmap(cm,annot=True,fmt="d")    
    print(classification_report(y_test,y_predict))

executeSVCModel(X_train, X_test)

#improving the model
min_train = X_train.min()
min_train
range_train = (X_train - min_train).max()
range_train

X_train_scaled = (X_train - min_train)/range_train

sns.scatterplot(x = X_train['Age'], y = X_train['401K Savings'], hue = y_train)

sns.scatterplot(x = X_train_scaled['Age'], y = X_train_scaled['401K Savings'], hue = y_train)


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


executeSVCModel(X_train_scaled, X_test_scaled)


#improving using GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)

grid.fit(X_train_scaled,y_train)

grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)

print(classification_report(y_test,grid_predictions))

