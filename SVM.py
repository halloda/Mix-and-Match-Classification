import numpy as np
import pandas


def costfn(Weight, Xi, Yi,Xtrain):
  N=Xtrain.shape[0]


  w2=np.dot(Weight,Weight) #the weight values squared 
  max_value=max(0,1-Yi*np.dot(Xi,W)) #b is stored in the weight vector 

  summation=np.sum(max_value)/N

  c_summation=alpha*summation #C times sigma

  cost= 1/2 * w2 + c_summation #full equation put together 

  return cost

  
    
def Cyixi(alpha, X, Y):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
  calc=alpha*np.dot(np.transpose(X),Y)

  return calc

#Derivative of cost fn
def gradient_fn(Weight, X, Y):

 
  numsamples=4
  newY=np.array(Y)
  deltaw=[]
  temp=0

 
  distance=1-(Y*np.dot(Weight,np.transpose(X))) 
  #print('dist:',distance.shape)
  for i in distance:

    if np.any(i,0)==0:
      temp=Weight
     # print("0 greater")
    else:
      temp=Weight-Cyixi(alpha,X,Y) 
      
    deltaw.append(temp)  

  deltaw=np.array(deltaw) 
  deltaw=sum(deltaw)
  fn=1/numsamples*deltaw
  
 # print("fn",fn)
  return fn



#Train using Stochastic Gradient Descent-- we want to minimize the cost fn to minimize misclassifications 


def sgd(features, outputs,Xtest,ytest,Xtrain):
  n=10
  niter=5
  lr=.01
  np_ones=np.ones((100778,1),dtype=int)
  new_weights=np.zeros([n])
  Xnew=Xtrain[:,1:n] #Our selected features for right now 
  
  Xnew=np.append(Xnew, np_ones, axis=1)

  for k in range(niter):
    prev_w=new_weights
    #print("prev_w",prev_w)
    new_weights=new_weights-lr*gradient_fn(prev_w,Xnew,outputs)
    #print("new w",new_weights)


  prediction=np.array([])
  Xnew_test=Xtest[:,1:n]
  np_test_ones=np.ones((25195,1),dtype=int)
  Xnew_test=np.append(Xnew_test,np_test_ones,axis=1)

  numpy_xtest=np.array(Xnew_test)
  for i in range(25195):
    temp_x=np.dot(new_weights,np.transpose(numpy_xtest[i]))
    #print(temp_x)
    testing_model=np.sign(temp_x) 
      
    prediction=np.append(prediction,testing_model)

  print("Weights: ",new_weights)
  print (list(prediction))
  print (list(ytest))

  from sklearn.metrics import accuracy_score

  acc=accuracy_score(ytest, prediction, normalize=True, sample_weight=None)
  print("Accuracy: ",acc)
  #use pyhthons built in accuracy calculation to compare y test to our prediction 

  from sklearn.metrics import precision_score
  prec=precision_score(ytest, prediction, average='weighted')
  print("Precision: ", prec)


#training the SVM 

alpha=.01


#call W=sgd() when chosen for classification
