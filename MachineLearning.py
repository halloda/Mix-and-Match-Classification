import tensorflow as tf
from tensorflow import keras



def RandomForest(Xtrain,Ytrain,Xtest,Ytest, FS, ML):
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn import metrics

    forest=RFC(n_estimators=50)
    forest.fit(Xtrain, Ytrain)


    newprediction=forest.predict(Xtest)

    currForestTest=Ytest

    print("Accuracy: ", metrics.accuracy_score(currForestTest,newprediction))

    print("Precision: ", metrics.precision_score(currForestTest,newprediction))
    
    TP,TN,FP,FN=0

    for i in range(len(newprediction)):
        if currForestTest[i]==newprediction[i]:
            TP+=1
        if newprediction[i]==1 and currForestTest[i]!=newprediction[i]:
            FP+=1
        if newprediction[i]==0 and currForestTest[i]!=newprediction[i]:
            FN+=1
        if currForestTest[i]==0 and newprediction[i]==0:
            TN+=1

    print ("TP: ", TP, " TN: ", TN)
    print ("FP ", FP, " FN ", FN)


def NeuralNetwork(Xtrain,Ytrain,Xtest,Ytest, FS, ML):
    import tensorflow as tf
    from tensorflow import keras

    nnmodel=keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(35, activation=tf.nn.relu),
    keras.layers.Dense(35, activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid),
    ])

    nnmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()])

    nnmodel.fit(Xtrain,Ytrain, epochs=1, batch_size=1)

    test_loss, test_acc, nnprec,nnfp, nnfn, nntn, nntp = nnmodel.evaluate(Xtest,Ytest)