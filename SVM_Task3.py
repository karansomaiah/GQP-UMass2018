
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import tarfile
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import argparse
import os.path as op
from os.path import join as pjoin


# In[7]:

def getAllFilesOfAUserInDirectory():
    
    parser = argparse.ArgumentParser(description='Getting the User ID')
    parser.add_argument("UserID", help='1st argument is the Users 1st session')
    parser.add_argument("dir", help='2nd argument is the Directory in which the files are located')
    
    args = parser.parse_args()
    
    #Reading all files from the directory
    #allFilesInDirectory = os.listdir("/home/abhishek/Downloads/GQP/Folder")
    allFilesInDirectory = os.listdir(str(args.dir))
    
    
    #Getting all files of User whose ID = 37
    #allFiles = [s for s in allFilesInDirectory if args.first in s]
    allFiles = [s for s in allFilesInDirectory if args.UserID in s]
    
    return allFiles, str(args.dir)
    

def readFilesAndImplementSVM(allFiles, directory):    
    dataFile = []
    
    #For all files, extracting the tar.gz into a pandas csv
    for fileNames in range(len(allFiles)):
        
        print("Reading and extracting file", allFiles[fileNames])
        
        fileLoc = directory + str(allFiles[fileNames])
        
        #Opening the tar.gz file
#         tar = tarfile.open(fileLoc, "r:gz")
        
#         #This loop iterates over tar files
#         for member in tar.getmembers():
            
#             #Extracts the tar.gz file
#             f = tar.extractfile(member)
            
            #Reads the tar.gz file into a pandas dataframe
        data = pd.read_csv(fileLoc, sep = ' ', header=None)

        firstThirthySeconds = data.iloc[0:61440]
        lastThirtySeconds = data.iloc[len(data)-61441:len(data)-1]

        firstThirthySeconds['Y_Variable'] = np.repeat(0,len(firstThirthySeconds))
        lastThirtySeconds['Y_Variable'] = np.repeat(1,len(lastThirtySeconds))

        df = firstThirthySeconds.append(lastThirtySeconds)

        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,2:130], df.iloc[:,130], test_size=0.20, random_state=42)    

        trainAccuracy, testAccuracy = svmImplementation(X_train, X_test, y_train, y_test, allFiles[fileNames])


def svmImplementation(X_train, X_test, y_train, y_test, tar):
    
    print("Implementing SVM")

    clf = svm.SVC()
    clf.fit(X_train,y_train)
    print("Predicting the train data")
    
    trainAccuracy = clf.score(X_train,y_train)
    print("Train accuracy =",trainAccuracy)
    
    print("Predicting the test data")
    
    testAccuracy = clf.score(X_test,y_test)
    print("Test accuracy =",testAccuracy)
    

    #Fitting decision function for Train and Test data
    print("Fitting the decision function for train and test data")
    decisionFuncValsTrain = clf.decision_function(X_train)
    decisionFuncValsTest = clf.decision_function(X_test)
    
    #Writing the decision function values to csv
    
    print("Writing the decision fuction values to csv")
    decisionFunctionValuesTrainData = pd.DataFrame(decisionFuncValsTrain,columns=['Dec_Values'])    
    fileNameForDecisionFunctionTrain = 'Output/'+str(tar)+'_decisionFunctionValuesTrainData.csv'
    decisionFunctionValuesTrainData.to_csv(fileNameForDecisionFunctionTrain)
    
    
    decisionFunctionValuesTestData = pd.DataFrame(decisionFuncValsTest,columns=['Dec_Values'])
    fileNameForDecisionFunctionTest = 'Output/'+str(tar)+'_decisionFunctionValuesTestData.csv'
    decisionFunctionValuesTestData.to_csv(fileNameForDecisionFunctionTest)

    
    decValuesTrain = np.array(decisionFuncValsTrain)
    decValuesTest = np.array(decisionFuncValsTest)

    #Finding the medians of train, test and whole data at a time for both classes
    print("Calculating median")
    medianOfClassATrainData = np.median(np.select(decValuesTrain > 0, decValuesTrain))
    medianOfClassATestData = np.median(np.select(decValuesTest > 0, decValuesTest))

    medianOfClassBTrainData = np.median(np.select(decValuesTrain < 0, decValuesTrain))
    medianOfClassBTestData = np.median(np.select(decValuesTest < 0, decValuesTest))
    
    print("Median of Class A for train Data =",medianOfClassATrainData)
    print("Median of Class B for train Data =",medianOfClassBTrainData)
    
    print("Median of Class A for test Data =",medianOfClassATestData)
    print("Median of Class B for test Data =",medianOfClassBTestData)
    
    mergedDfOfDecisionFnValues = decisionFunctionValuesTrainData.append(decisionFunctionValuesTestData)
    medianOfClassA = np.median(np.select(np.array(mergedDfOfDecisionFnValues) > 0, np.array(mergedDfOfDecisionFnValues)))
    medianOfClassB = np.median(np.select(np.array(mergedDfOfDecisionFnValues) < 0, np.array(mergedDfOfDecisionFnValues)))

    print("Median of Class A for whole data =",medianOfClassA)
    print("Median of Class B for whole Data =",medianOfClassB)
    
    filename = str(tar) + '.txt'
    path_to_file = pjoin("Output", filename)
    FILE = open(path_to_file, "w")
    
    trainAccWr = 'Train Accuracy =' + str(trainAccuracy)
    testAccWr = 'Test Accuracy =' + str(testAccuracy)
    medianOfClassATrainDataWr = 'Median of Class A for train Data =' + str(medianOfClassATrainData)
    medianOfClassATestDataWr = 'Median of Class A for train Data =' + str(medianOfClassATestData)
    medianOfClassBTrainDataWr = 'Median of Class A for train Data =' + str(medianOfClassBTrainData)
    medianOfClassBTestDataWr = 'Median of Class A for train Data =' + str(medianOfClassBTestData)
    medianOfClassAWr = 'Median of Class A for whole data =' + str(medianOfClassA) + '\n'
    medianOfClassBWr = 'Median of Class B for whole data =' + str(medianOfClassB) + '\n'
    

    #FILE.write(trainAccWr)
    #FILE.write(testAccWr)
    #FILE.write(medianOfClassATrainDataWr)
    #FILE.write(medianOfClassATestDataWr)
    #FILE.write(medianOfClassBTrainDataWr)
    #FILE.write(medianOfClassBTestDataWr)
    FILE.write(medianOfClassAWr)
    FILE.write(medianOfClassBWr)
    
    FILE.close()
    
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     gammas = [0.001, 0.01, 0.1, 1]
#     #kernel = ['linear', 'poly', 'rbf', 'sigmoid']
#     param_grid = {'C': Cs, 'gamma' : gammas}
#     grid_search = gs.GridSearchCV(svm.SVC(kernel='rbf'), param_grid,verbose=1)
#     grid_search.fit(X_train, y_train)
#     print("Best parameters for this dataset are",grid_search.best_params_)
    
#     print("Predicting the train data")
#     y_pred = grid_search.predict(X_train)
#     print("Calculating Train Accuracy")
#     trainAccuracy = (np.sum(y_pred == y_train)/len(y_train))*100
#     print("Train accuracy =",trainAccuracy)
    
#     print("Predicting the train data")
#     y_test_pred = grid_search.predict(X_test)
#     print("Calculating Test Accuracy")
#     testAccuracy = (np.sum(y_test_pred == y_test)/len(y_test))*100
#     print("Test accuracy =",testAccuracy)
    

    return trainAccuracy, testAccuracy


# In[8]:

def main():
    allFiles, directory = getAllFilesOfAUserInDirectory()
    readFilesAndImplementSVM(allFiles, directory)

if __name__ == "__main__":
    main()

