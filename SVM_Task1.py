
# coding: utf-8

# In[4]:

#Coded by:
#SVM Implementation 


# In[1]:

import pandas as pd
import numpy as np

#To read the tar.gz file
import tarfile

#To locate all files in a directory
import os

#Scikit learn's svm and train/test split imports
from sklearn import svm
from sklearn.model_selection import train_test_split

#To read parameter values from the command line
import argparse


# In[3]:

def getAllFilesOfAUserInDirectory():
    
    parser = argparse.ArgumentParser(description='Getting the User ID')
    parser.add_argument("User_ID", help='1st argument is the Users 1st session')
    args = parser.parse_args()
    
    #Reading all files from the directory
    allFilesInDirectory = os.listdir("/home/abhishek/Downloads/GQP/Folder")

    #Getting all files of User whose ID is given as a parameter in the command line
    allFiles = [s for s in allFilesInDirectory if args.User_ID in s]
    return allFiles, args.User_ID
    

def readFiles(allFiles):    
    dataFile = []
    
    #For all files, extracting the tar.gz into a pandas csv
    for fileNames in range(len(allFiles)):
        
        print("Reading and extracting file", allFiles[fileNames])
        
        #Opening the tar.gz file
        tar = tarfile.open(allFiles[fileNames], "r:gz")
        
        #This loop iterates over tar files
        for member in tar.getmembers():
            
            #Extracts the tar.gz file
            f = tar.extractfile(member)
            
            #Reads the tar.gz file into a pandas dataframe
            data = pd.read_csv(f, sep = ' ', header=None, nrows=61440)
            data['Y_Variable'] = np.repeat(fileNames,len(data))
            
            #Appending the dataframe values to a list
            dataFile.append(data)
            
    #Convering the list into a pandas dataframe
    df = pd.DataFrame()
    df = pd.concat(dataFile)
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,2:130], df.iloc[:,136], test_size=0.20, random_state=42)    
    
    return X_train, X_test, y_train, y_test


def svmImplementation(X_train, X_test, y_train, y_test):
    
    print("Implementing SVM")
    
    #Calling scikit's SVC function, with default values
    clf = svm.SVC(verbose=1)
    
    #Fitting the model on the train data
    clf.fit(X_train,y_train)
    print("Predicting the train data")
    
    #Predicting values for train data
    #y_pred = clf.predict(X_train)
    #print("Calculating Train Accuracy")
    
    #Calculating the train accuracy by comparing actual values to the predicted values
    #trainAccuracy = (np.sum(y_pred == y_train)/len(y_train))*100
    
    trainAccuracy = clf.score(X_train,y_train)
    print("Train accuracy =",trainAccuracy)
    
    #Predicting values for the test data
    #y_test_pred = clf.predict(X_test)
    #print("Calculating Test Accuracy")
    
    #Calculating the test accuracy by comnparing actual values to the predicted values
    #testAccuracy = (np.sum(y_test_pred == y_test)/len(y_test))*100
    #print("Test accuracy =",testAccuracy)
    
    testAccuracy = clf.score(X_test,y_test)
    print("Test accuracy =",testAccuracy)
    
    decisionFunctionValues = pd.DataFrame(clf.decision_function(X_train))
    
    decisionFunctionValues.to_csv("DecisionFunctionValues.csv")
    
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     #kernel = ['linear', 'poly', 'rbf', 'sigmoid']
#     param_grid = {'C': Cs, 'gamma' : gammas}
#     grid_search = gs.GridSearchCV(svm.SVC(kernel='rbf',max_iter=1000), param_grid,verbose=1)
#     grid_search.fit(X_train, y_train)
#     print(grid_search.best_params_)

    return trainAccuracy, testAccuracy
    
def writeToFile(trainAccuracy, testAccuracy,userID,allFiles):
    df = pd.DataFrame(columns=['Train', 'Test','User_ID'])
    df['Train'] = trainAccuracy
    df['Test'] = testAccuracy
    df['User_ID'] = userID
    #df['File_Names'] = allFiles
    fileName = str(userID) + '.csv'
    df.to_csv(fileName, sep=',')


# In[92]:

def main():
    allFiles, userID = getAllFilesOfAUserInDirectory()
    X_train, X_test, y_train, y_test = readFiles(allFiles)
    trainAccuracy, testAccuracy = svmImplementation(X_train, X_test, y_train, y_test)
    #writeToFile(trainAccuracy,testAccuracy,userID,allFiles)

if __name__ == "__main__":
    main()

