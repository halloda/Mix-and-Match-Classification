import numpy as np
import pandas
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler 

from FeatureSelection import *
from MachineLearning import *


def Mushrooms(DS,FS,ML):
    full_dataset=pandas.read_csv('mushrooms.csv')
    full_dataset.replace([np.inf, -np.inf], np.nan)
    full_dataset.fillna(0)

    target=full_dataset['class'].copy()
    full_dataset.drop('class', axis=1, inplace=True)

    enc_df=pandas.get_dummies(full_dataset) #convert categorical to dummy variables
    enc_df.fillna(0)

    target[target=='p']=1   #numerical data needed for RF and NN
    target[target=='e']=0

#-----------Feature Selection Choice-------------
    if FS=='Kbest':
        newX=Kbest(enc_df,target) 
    elif FS=='Model':
        newX=SelectModel(enc_df,target) #Linear Regression
#-------------------------------------------------
    Xtrain, Xtest,Ytrain,Ytest=train_test_split(newX, target, test_size=.2,random_state=42)

    Xtrain=np.asarray(Xtrain)
    Ytrain=np.asarray(Ytrain)
    Xtest=np.asarray(Xtest)
    Ytest=np.asarray(Ytest)

    Xtrain=Xtrain.astype(int)
    Ytrain=Ytrain.astype(int)
    Xtest=Xtest.astype(int)
    Ytest=Ytest.astype(int)     #format for sklearn
    


#-----------Machine Learning Choice-----------------
    if ML=='Forest':
        RandomForest(Xtrain,Ytrain,Xtest,Ytest,FS,ML)
    elif ML=='NeuralNet':
        NeuralNetwork(Xtrain,Ytrain,Xtest,Ytest,FS,ML)
#---------------------------------------------------
    
    
def Network(DS,FS,ML):

    full_dataset=arff.loadarff('KDDTrain+.arff')

    trdf=pandas.DataFrame(full_dataset[0])
    enc_tr_df=pandas.get_dummies(trdf)
    enc_tr_df.fillna(0)

    Xtotal=enc_tr_df.iloc[:,0:120]
    ytotal=enc_tr_df["class_b'anomaly'"]

    #-----------Feature Selection Choice-------------
    if FS=='Kbest':
        newX=Kbest(Xtotal,ytotal) 
    elif FS=='Model':
        newX=SelectModel(Xtotal,ytotal)
    #-------------------------------------------------

    Xtrain,Xtest,ytrain,ytest=train_test_split(newX,ytotal,test_size=0.2)

    ytrain=np.array(ytrain)
    ytrain=ytrain.astype(int)
    

    ytest=np.array(ytest)
    ytest=ytest.astype(int)
    

    #print("X train: ", Xtrain.shape, "Y: ", ytrain.shape)
    #print("X: ", Xtest.shape, "Y: ", ytest.shape)

#-----------Machine Learning Choice-----------------
    if ML=='Forest':
        RandomForest(Xtrain,ytrain,Xtest,ytest,FS,ML)
    elif ML=='NeuralNet':
        NeuralNetwork(Xtrain,ytrain,Xtest,ytest,FS,ML)
#---------------------------------------------------



def dataSetChoice(DS,FS,ML):
    if DS=='Mushrooms':
        Mushrooms(DS,FS,ML)
    elif DS=='Network':
        #Network(DS,FS,ML)
        print("test")


#dataSetChoice(DS,FS,ML)