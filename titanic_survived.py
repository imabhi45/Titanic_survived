# dataset taken from kaggle

import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
import matplotlib.pyplot as pplt

#training dataset
training_data = pd.read_csv('train.csv')
training_data = training_data.dropna()
training_data = training_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

#test dataset
test_data = pd.read_csv('test.csv')
test_data =test_data.fillna(0)
z = test_data['PassengerId']
test_data = test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1)

#change gender in datasets
training_data['Sex'] = training_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#take output and input from training dataset
y = training_data['Survived']
x = training_data.drop('Survived', axis=1)

#create model
model = lr()
model.fit(x, y)

#prediction from model 
y_predict = model.predict(test_data)

#to find how accuracy of model
print("From model ",model.score(x, y))

#plot graph
pplt.scatter(test_data, y_predict)
pplt.show()
