
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import tarfile
import os
from sklearn.model_selection import train_test_split
import argparse
import os.path as op
from os.path import join as pjoin
from sklearn.neighbors.nearest_centroid import NearestCentroid
import csv
import time


# In[2]:

def getAllFilesOfAUserInDirectory():
    
    parser = argparse.ArgumentParser(description='Getting the User ID')
    parser.add_argument("UserID", help='1st argument is the Users 1st session')
    parser.add_argument("dir", help='2nd argument is the Directory in which the files are located')
    parser.add_argument("FFT", help='3rd argument is whether to run FFT or not')
    
    
    
    args = parser.parse_args()
    
    #Reading all files from the directory
    #allFilesInDirectory = os.listdir("/home/abhishek/Downloads/GQP/Folder")
    allFilesInDirectory = os.listdir(str(args.dir))
    
    
    #Getting all files of User whose ID = 37
    #allFiles = [s for s in allFilesInDirectory if args.first in s]
    allFiles = [s for s in allFilesInDirectory if args.UserID in s]
    
    return allFiles, str(args.dir), str(args.UserID), str(args.FFT)
    

def readFilesAndImplementNearestCentroid(allFiles, directory, userID, FFT):    
    dataFile = []
    
    #For all files, extracting the tar.gz into a pandas csv
    for fileNames in range(len(allFiles)):
        
        print("Reading and extracting file", allFiles[fileNames])
        
        fileLoc = directory + str(allFiles[fileNames])
        
        #Opening the tar.gz file
#        tar = tarfile.open(fileLoc, "r:gz")
        
        #This loop iterates over tar files
#         for member in tar.getmembers():
            
#             #Extracts the tar.gz file
#             f = tar.extractfile(member)
            
            
            #Reads the tar.gz file into a pandas dataframe
        data = pd.read_csv(fileLoc, sep = ' ', header=None, skiprows=1, usecols=list(range(2, 130)), dtype=np.float32)
        
#        f.close()

        #Reads the first and last thirty seconds of data
        firstThirthySeconds = data.iloc[0:61440]
        lastThirtySeconds = data.iloc[len(data)-61441:len(data)-1]

        #Assigns label to each class (baseline vs meditation for first 30 vs last 30 seconds)
        firstThirthySeconds['Y_Variable'] = np.repeat(0,len(firstThirthySeconds))
        lastThirtySeconds['Y_Variable'] = np.repeat(1,len(lastThirtySeconds))

        #Appending into a single dataframe
        df = firstThirthySeconds.append(lastThirtySeconds)

        x_classA = df.iloc[0:61440,0:128]
        x_classB = df.iloc[61440:61440*2,0:128]

        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:128], df.iloc[:,128], test_size=0.10, random_state=42)    

        #NearestCentroidImplementation(X_train, X_test, y_train, y_test, x_classA, x_classB, userID)
        
        if FFT =='Yes':
            implementFFT(X_train, X_test, y_train, y_test, x_classA, x_classB, userID)
        else:
            NearestCentroidImplementation(X_train, X_test, y_train, y_test, x_classA, x_classB, userID)
        
        print("Successfully processed the dataset")

            
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
def NearestCentroidImplementation(X_train, X_test, y_train, y_test, x_classA, x_classB, userID):
    
    print("Implementing Nearest Centroid")

    clf = NearestCentroid()
    clf.fit(X_train,y_train)
    print("Predicting the train data")
    
    trainAccuracy = clf.score(X_train,y_train)
    print("Train accuracy =",trainAccuracy)
    
    print("Predicting the test data")
    
    testAccuracy = clf.score(X_test,y_test)
    print("Test accuracy =",testAccuracy)
    
    #Getting the centroid of Class A and Class B
    centroids = clf.centroids_
    centroidClassA = np.array(centroids[0])
    centroidClassB = np.array(centroids[1])

    
    distanceA, distanceB = distForIndivClassesFromCentroid(x_classA, centroidClassA, centroidClassB, x_classB, userID)
    
    analysisClassWisePointsToCentroid(x_classA, centroidClassA, centroidClassB, x_classB, userID, distanceA, distanceB)
    
    analysisAllPointsToBothCentroids(x_classA, centroidClassA, centroidClassB, x_classB, userID, distanceA, distanceB)

def analysisClassWisePointsToCentroid(x_classA, centroidClassA, centroidClassB, x_classB, userID, distanceA, distanceB):
    
    print("Implementing weighted distance for class-wise points from centroids")
    
    allOrIndiv = 'indiv'
    
    weightedAverageCalculation(distanceA, distanceB, userID, allOrIndiv)
    
    
def analysisAllPointsToBothCentroids(x_classA, centroidClassA, centroidClassB, x_classB, userID, distanceA, distanceB):
    
    print("Implementing weighted distance for all points from the Centroids")
    
    #Assigning an indicator to the weighted average function
    allOrIndiv = 'all'
    
    disA = distanceA
    disB = distanceB
    
#     disA = dataNormalizer(disA).tolist()
#     disB = dataNormalizer(disB).tolist()
    
    for i in range(len(x_classA)):    
        distA = np.linalg.norm(centroidClassA-x_classB.iloc[i])
        disA.append(distA)
        
    for j in range(len(x_classB)):
        distB = np.linalg.norm(centroidClassB-x_classA.iloc[i])
        disB.append(distB)
    
    weightedAverageCalculation(disA, disB, userID, allOrIndiv)
        
def distForIndivClassesFromCentroid(x_classA, centroidClassA, centroidClassB, x_classB, userID):
    
    print("Calculating distances from centroid to the two classes")
    
    distanceA = []
    distanceB = []
    
    #Calculating the distance from the centroid of class A to all elements of class A and similarly for class B
    for i in range(len(x_classA)):    
        distA = np.linalg.norm(centroidClassA-x_classA.iloc[i])
        distanceA.append(distA)
    
    for i in range(len(x_classA)):    
        distB = np.linalg.norm(centroidClassB-x_classB.iloc[i])
        distanceB.append(distB)
        
    distanceClassA = np.array(distanceA)
    distanceClassB = np.array(distanceB)
    
#     distanceClassA = dataNormalizer(distanceClassA)
#     distanceClassB = dataNormalizer(distanceClassB)
    
    filenameMean = 'DistanceStatsMean_NearestCentroid.csv'
    filenameMedian = 'DistanceStatsMedian_NearestCentroid.csv'
    filenameMax = 'DistanceStatsMax_NearestCentroid.csv'
    
    path_to_mean_file = pjoin("Output", filenameMean)
    path_to_median_file = pjoin("Output", filenameMedian)
    path_to_max_file = pjoin("Output", filenameMax)
    
    
    print("Obtaining the mean, median and max distance from centroid to the classes")
    distanceMean = [userID, np.mean(distanceClassA), np.mean(distanceClassB)]
    distanceMedian = [userID, np.median(distanceClassA), np.median(distanceClassB)]
    distanceMax = [userID, np.max(distanceClassA), np.max(distanceClassB)]
    
    print("Writing the mean, median and max to file")
    
    with open(path_to_mean_file,'a') as f:
        wtr = csv.writer(f)
        wtr.writerow(distanceMean)
        f.close()
    
    with open(path_to_median_file,'a') as f:
        wtr = csv.writer(f)
        wtr.writerow(distanceMedian)
        f.close()
    
    with open(path_to_max_file,'a') as f:
        wtr = csv.writer(f)
        wtr.writerow(distanceMax)
        f.close()
        
        
    return distanceA, distanceB

def weightedAverageCalculation(distClassA, distClassB, userID, allOrIndiv):
    
    weightsClassA = []
    weightsClassB = []
    
    mu = 0
    sigA = np.std(np.array(distClassA))
    sigB = np.std(np.array(distClassB))
    
    print("Finding the weighted average for class A")
    for i in range(len(distClassA)):
        weightsClassA.append(gaussian(distClassA[i],mu,sigA))
        
    print("Finding the weighted average for class B")
    for j in range(len(distClassB)):
        weightsClassB.append(gaussian(distClassB[i],mu,sigB))
        
    
    weightedAverageClassA = np.average(distClassA, weights=weightsClassA)
    weightedAverageClassB = np.average(distClassB, weights=weightsClassB)
    
    if (allOrIndiv == 'all'):
        
        print("Writing weighted averages for all points to file")
    
        fileNameWeightedDistances = 'WeightedDistancesForAllPoints_NearestCentroid.csv'
    
    else:
        
        print("Writing weighted averages of class-wise points to file")
        
        fileNameWeightedDistances = 'WeightedDistancesForIndivClasses_NearestCentroid.csv'
    
    path_to_weighted_file = pjoin("Output", fileNameWeightedDistances)
    
    distanceWeighted = [userID, weightedAverageClassA, weightedAverageClassB]
    
    with open(path_to_weighted_file,'a') as f:
        wtr = csv.writer(f)
        wtr.writerow(distanceWeighted)
        f.close()
    
def dataNormalizer(x):
    scaledValues = []
    minVal = np.min(x)
    maxVal = np.max(x)
    for i in range(len(x)):
        scaledValues.append((x[i] - minVal) / (maxVal - minVal))  
    return np.array(scaledValues)


def implementFFT(X_train, X_test, y_train, y_test, x_classA, x_classB, userID):
    
    fft_X_train = np.fft.fft(X_train)
    phase_train = np.angle(fft_X_train)
    amplitude_train = np.absolute(fft_X_train)
    
    fft_X_test = np.fft.fft(X_test)
    phase_test = np.angle(fft_X_test)
    amplitude_test = np.absolute(fft_X_test)
    
    #print(amplitude_train[0])
    #print(amplitude_test[0])
    
    NearestCentroidImplementation(phase_train, phase_test, y_train, y_test, x_classA, x_classB, userID)
    
    


# In[ ]:

def main():
    allFiles, directory, userID, FFT = getAllFilesOfAUserInDirectory()
    readFilesAndImplementNearestCentroid(allFiles, directory, userID, FFT)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))


# In[34]:

#data = pd.read_csv('/home/abhishek/Downloads/GQP/raw_data_3task/ID37_S05_BA_05_110916_1253_Run1_raw.txt', sep=' ', nrows=10, header=None, usecols=list(range(2, 130)), dtype=np.float32)


# In[37]:

#data.head()


# In[8]:

# import numpy.fft
# fft_X_train = np.fft.fft(data)
# phase = np.angle(fft_X_train)
# amplitude = np.absolute(fft_X_train)


# In[7]:

#phase


# In[9]:

#amplitude


# In[20]:

#list(range(2, 13))


# In[ ]:



