import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading the data from CSV file to a Pandas DataFrame
parkinsons_data = pd.read_csv ("C:\\Users\\DSR\\PycharmProjects\\pythonProject1\\\parkinsons.csv")
#parkinsons_data.head()
X = parkinsons_data.drop(columns = ['name','status'], axis = 1)
Y = parkinsons_data['status']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.4,random_state=5)
# Visualising the Original/Train/Test sizes
print ( X.shape,X_train.shape,X_test.shape)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = svm.SVC(kernel = 'linear')
#training the svm model with the training data
model.fit(X_train,Y_train)
# Accuracy Score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print ("Training Data Accuracy Score = ",training_data_accuracy)
# Accuracy Score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print ("Test Data Accuracy Score = ",test_data_accuracy)


model1 = XGBClassifier()
#training the svm model with the training data
model1.fit(X_train,Y_train)
# Accuracy Score on training data
X_train_prediction1 = model1.predict(X_train)
training_data_accuracy1 = accuracy_score(Y_train, X_train_prediction1)
print ("Training Data Accuracy Score = ",training_data_accuracy1)
# Accuracy Score on test data
X_test_prediction1 = model1.predict(X_test)
test_data_accuracy1 = accuracy_score(Y_test, X_test_prediction1)
print ("Test Data Accuracy Score = ",test_data_accuracy1)


model2 = KNeighborsClassifier(n_neighbors=3)
#training the svm model with the training data
model2.fit(X_train,Y_train)
# Accuracy Score on training data
X_train_prediction2 = model2.predict(X_train)
training_data_accuracy2 = accuracy_score(Y_train, X_train_prediction2)
print ("Training Data Accuracy Score = ",training_data_accuracy2)
# Accuracy Score on test data
X_test_prediction2 = model2.predict(X_test)
test_data_accuracy2 = accuracy_score(Y_test, X_test_prediction2)
print ("Test Data Accuracy Score = ",test_data_accuracy2)


model3 = RandomForestClassifier(random_state=0)
#training the svm model with the training data
model3.fit(X_train,Y_train)
# Accuracy Score on training data
X_train_prediction3 = model3.predict(X_train)
training_data_accuracy3 = accuracy_score(Y_train, X_train_prediction3)
print ("Training Data Accuracy Score = ",training_data_accuracy2)
# Accuracy Score on test data
X_test_prediction3 = model3.predict(X_test)
test_data_accuracy3 = accuracy_score(Y_test, X_test_prediction3)
print ("Test Data Accuracy Score = ",test_data_accuracy3)






#Building a predictive system
#Data of a parkinson's Disease patient
input_data = (197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569)
#Changing input data to a numpy array
input_data_as_array = np.array(input_data)

#Reshape the numpy array
input_data_reshaped = input_data_as_array.reshape(1,-1)

#Standarizing the Data
std_data = scaler.transform(input_data_reshaped)
std_data1 = scaler.transform(input_data_reshaped)
std_data2 = scaler.transform(input_data_reshaped)
std_data3 = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
prediction1 = model1.predict(std_data1)
prediction2 = model2.predict(std_data2)
prediction3 = model3.predict(std_data3)

print ("The Accuracy Prediction of Disease by using SVM is = ", prediction)
print ("The Accuracy Prediction of Disease by using XGB is = ", prediction1)
print ("The Accuracy Prediction of Disease by using KNN is = ", prediction2)
print ("The Accuracy Prediction of Disease by using RFA is = ", prediction3)

accuracy = (((prediction+prediction1+prediction2+prediction3))/4)*100
print("Average Accuracy of Parkinson Disease =",accuracy )
if(accuracy>50):
    print("The person has parkinson Disease ")
else:
    print("The person has no parkinson Disease")

