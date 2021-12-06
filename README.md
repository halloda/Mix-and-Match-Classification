Mix and Match Classification Application

Overview
The Mix and Match Classification App (MaMC) allows you to make different choices with your data to streamline the machine learning classification process. 
For example, if you need to classify network intrusion data, you can choose the type of feature selection and the type of algorithm you want to use and compare the results. These results will ensure that we find the combination that best classifies the data as intrusion or normal data. 

Adding a Dataset 
The MaMC App connects to data sets via the pandas python library in the Datasets.py file. It currently can accept files with the .csv and .arff extension.  
Let’s add a new data file following these steps:
1.	Upload the csv or arff file into the main MaMC folder
2.	Define a new function in the Dataset.py file following the code samples using csv and arff files. Note the DS, FS, and ML variables being passed, these allow the application to keep track of and communicate which dataset, feature selection, and machine learning algorithm is chosen.


 ![image](https://user-images.githubusercontent.com/44102740/144881407-23ca2909-26c4-48ef-aabe-9c85653c6be0.png)
![image](https://user-images.githubusercontent.com/44102740/144881440-6cbd426a-4e16-4bf7-8e03-1a0eb7ff6a97.png)

3.	Continue to clean the dataset using the pandas library to convert categorical data (categories such as colors, visual features, or names) into numerical data so that the machine learning algorithms can easily read the data. Fill in any empty data values with the fillna() function. 


![image](https://user-images.githubusercontent.com/44102740/144881507-7f61aed5-aa93-4f0c-9584-62819d05eed3.png)

4.	Next, we will need to add the feature selection capability into the current function. To do this, we need a simple if/else if statement to allow the ability of feature selection to run on the new data. 


![image](https://user-images.githubusercontent.com/44102740/144881553-359d2909-1108-4fe9-bfc5-23f29606518e.png)

5.	Once feature selection is added to the dataset, we can divide the data into training and testing datasets using the train_test_split() function before using it in a machine learning algorithm.
6.	After the data is split, we can add the ability to choose a machine learning algorithm. This is another easy if/else if statement to enable this functionality.
  ![image](https://user-images.githubusercontent.com/44102740/144881585-23dff597-cca4-42da-8bc3-62b75f22a06f.png)


Adding Feature Selection 
To add new Feature Selection methods in the MaMC App, you will need to access and edit the FeatureSelection.py file. 
1.	Define a new function of the feature selection method you want to implement. You can write an algorithm from scratch or use an algorithm from a python library. 
2.	Add the newly defined function to the feature selection section to the Dataset.py file if/else if choices.

Adding a Machine Learning Algorithm 
To add a new machine learning algorithm in the MaMC App, you will need to access and edit the MachineLearning.py file. 
1.	Define a new function of the machine learning algorithm you are wanting to use. You can write an algorithm from scratch or use an algorithm from a python library. 
2.	Add the newly defined function to the machine learning algorithm section to the Dataset.py file if/else if choices. 

Running the Application
The MaMC Application uses a simple interface that allows you to first choose a dataset from the previously uploaded and programmed options. You can then choose from the available feature selection methods that will find the best categories to use for classifying your data.
Once those choices are made, a machine learning classification algorithm is chosen to run on the data. Metrics of the accuracy and success of the machine learning algorithm based on your data and feature selection choices will then be displayed. 

![image](https://user-images.githubusercontent.com/44102740/144881193-89ab52f7-6ca8-417f-b58d-fc778f0e808c.png)


