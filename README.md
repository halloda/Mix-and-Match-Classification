# Mix-and-Match-Classification
Choose between multiple data sets, feature selection methods, and algorithms to create classification. 

Through the interface file the user chooses their selection of data set, feature selection method, and classification method. 

![image](https://user-images.githubusercontent.com/44102740/128958261-fed9dc58-882c-4ae8-bbca-39c8c69aa395.png)

Then, the matching data set function in the dataset.py file is called and run. This function reads in the data, fills in missing values, and changes data to numerical for processing into the feature selection and machine learning functions. 

Once the data is run through, the feature selection method is run, the data is then split into testing and training sets, and then ran through the user selected classification algorithm. 


