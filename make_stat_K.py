import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas
import csv

data = pandas.read_csv('titanic\\train.csv')

dataX = data[['Pclass', 'Sex']]
dataX = dataX.dropna()  # delete all raws with Nan
dataX.Sex.replace(['male', 'female'], [1, 0], inplace=True)  # replace 'cause need of int, not str
y = data.Survived
##########
## cheat to know, what params are better
## but now doesnt work :(
#alg = DecisionTreeClassifier(random_state=1)
#params = [{
# "min_samples_split": [2, 4, 6, 8, 10], # any numbers to check
# "min_samples_leaf": [1, 2, 4] # same
}]
# grid = GridSearchCV(alg, params, refit=True, n_jobs=-1)
# grid.fit(dataX, y)
# alg_best = grid.best_estimator_
# alg.fit(dataX, y)
# print(alg.score(dataX, y))
# print(f"Accuracy: {alg.best_score_}  with params {grid.best_params_}")
###############

clf = DecisionTreeClassifier(random_state=1, min_samples_split=2, min_samples_leaf=1)
clf.fit(dataX, y)  # обучили
test_data = pandas.read_csv('titanic\\test.csv')  # берём тестовые данные

progn_data = test_data[['Pclass', 'Sex']]
progn_data.Sex.replace(['male', 'female'], [1, 0], inplace=True)
progn = clf.predict(progn_data)  # предсказываем
progn2 = pandas.DataFrame(progn, columns=['Survived'])
res = pandas.concat([progn_dat['PassengerId'], progn2['Survived']], names=['PassengerId', 'Survived'], axis='columns')

filename = 'res.csv'
res.to_csv('res.csv', index=False, encoding='utf-8')

res2 = pandas.read_csv('res.csv')
print(res2.columns.values)
print(res2)
