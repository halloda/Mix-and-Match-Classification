from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
#look over the last one for sure

def Kbest(X,Y):
    newFeatures=SelectKBest(f_classif, k=10).fit.transform(X,Y)
    return newFeatures

def SelectModel(X,Y):
    newFeatures = SelectFromModel(estimator=LogisticRegression(), max_features=3,  threshold=-np.inf)
    newFeatures = newFeatures.fit_transform(X, Y)
    return newFeatures