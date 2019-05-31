#%%
# import os
# current_path = os.getcwd()
# find = current_path.find('Ensemble')
# if find == -1:
#     os.chdir('./07-Ensemble/')
# os.getcwd()

#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler

#%%
data = pd.read_csv('/Users/pitichampeethong/website/casugol-abde/dataset/cancer.csv')
data.head()

#%%
data.drop(['Sample code number'],axis = 1, inplace = True)

#%%
data.head()

#%%
data.describe()

#%%
data.info()

#%%
data['Bare Nuclei']

#%%
data.replace('?',0, inplace=True)
data['Bare Nuclei']

#%%
# Convert the DataFrame object into NumPy array otherwise you will not be able to impute
values = data.values

# Now impute it
imputer = Imputer()
imputedData = imputer.fit_transform(values)

#%%
scaler = MinMaxScaler(feature_range=(0, 1))
normalizedData = scaler.fit_transform(imputedData)

#%%
# Bagged Decision Trees for Classification - necessary dependencies

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#%%
# Segregate the features from the labels
X = normalizedData[:,0:9]
Y = normalizedData[:,9]

#%%
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#%%
# AdaBoost Classification

from sklearn.ensemble import AdaBoostClassifier
seed = 7
num_trees = 70
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#%%
# Voting Ensemble for Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression(solver='lbfgs')
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC(gamma='scale')
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())