from mini import svm_read_problem
import numpy as np

"""
Loads LIBSVM dataset from a file and returns an np array for the data points of trainingData
"""
def loadData(fileName, threshhold):
    a = None
    w = None
    labelVector = None
    a = svm_read_problem(fileName, True)
    trainingData = a[1].toarray()
    shuffle = np.random.permutation(len(trainingData))
    trainingData = trainingData[shuffle]
    # print(trainingData)
    # K = initializeGram(trainingData)

    w = np.zeros(trainingData.shape[0]) #len(a[0]) is number of dimensions
    w = np.append(w,threshhold)
    return trainingData, a[0][shuffle], w #returns trainingData vector, label vector, and weight vector

features, label, w = loadData("/Users/jamescourson/Documents/COMP5680/mini-2/a4a.all", .8)
data = [features, label, w]

def split(p1, data): #percent of data to make training
    p1 = int(p1)
    if p1 > 99:
        print("percent is not an less than %99")
        pass
    index = int((len(data[0])*p1)/100)
    trainingData = [data[0][:index], data[1][:index], np.append(data[2][:index],data[2][-1])]
    testingData = [data[0][index:], data[1][index:], data[2][index:]]
    return trainingData, testingData


trainingData, testingData = split(70, data) # seems to work :D

# print(trainingData[0])
# print(trainingData[1])
# print(trainingData[2])
#
# print(testingData[0])
# print(testingData[1])
# print(testingData[2])
