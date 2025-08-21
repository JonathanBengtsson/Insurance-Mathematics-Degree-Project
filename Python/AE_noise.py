import numpy as np
import random
import copy

#Functions used for adding noise when training autoencoders
#----------------------------------------------------------------------------------

def createDistribution(samples):
    n = len(samples)
    m = len(samples[0])

    dist = list(np.zeros(m))

    for k in range(n): 
        for k2 in range(m):
            if samples[k][k2] == 1:
                dist[k2] += 1

    for k in range(m):
        dist[k] /= n

    #write as cumulative
    for k in range(m):
        if k>0:
            dist[k] += dist[k-1]

    return(dist)


def generateSample(dist):
    m = len(dist)
    r = random.uniform(0, 1)
    sample = list(np.zeros(m))
    n = 0

    for k in range(m):
        if k>0:
            if r > dist[k-1] and r <= dist[k]:
                n = k

    sample[n] = 1
    return(sample)

def randomIndices(nIndex, nTotal): #without replacement
    indicies = []

    allIndex = [*range(0, nTotal, 1)] 

    for k in range(nIndex):
        r = random.randint(0, len(allIndex)-1)
        indicies.append(allIndex[r])
        allIndex.pop(r)

    return(indicies)


#For categorical features
#------------------------
#For both type = 'sample' and type = 'null': 
#chose 'level' to the number of categories to be affected by noise for each sample.

def addPerturbationToCat(type, level, data1, data2, data3, data4, data5, data6): 
    data1Pert = list(data1)
    data2Pert = list(data2)
    data3Pert = list(data3)
    data4Pert = list(data4)
    data5Pert = list(data5)
    data6Pert = list(data6)
  
    if type == 'sample': 
        data1Dist = createDistribution(data1)
        data2Dist = createDistribution(data2)
        data3Dist = createDistribution(data3)
        data4Dist = createDistribution(data4)
        data5Dist = createDistribution(data5)
        data6Dist = createDistribution(data6)
        for k in range(len(data1)):          #all data has same length (data1 arbitrary here)
            picked = randomIndices(level, 6) #chose 'level' number of categories to be affected by noise for this sample
            for k2 in picked:
                if k2 == 0:
                    data1Pert.pop(k)
                    data1Pert.insert(k, generateSample(data1Dist))
                elif k2 == 1:
                    data2Pert.pop(k)
                    data2Pert.insert(k, generateSample(data2Dist))
                elif k2 == 2:
                    data3Pert.pop(k)
                    data3Pert.insert(k, generateSample(data3Dist))
                elif k2 == 3:
                    data4Pert.pop(k)
                    data4Pert.insert(k, generateSample(data4Dist))
                elif k2 == 4:
                    data5Pert.pop(k)
                    data5Pert.insert(k, generateSample(data5Dist))
                elif k2 == 5:
                    data6Pert.pop(k)
                    data6Pert.insert(k, generateSample(data6Dist))
    elif type == 'null':
        for k in range(len(data1)):          #all data has same length (data1 arbitrary here)
            picked = randomIndices(level, 6) #chose 'level' number of categories to be affected by noise for this sample
            for k2 in picked:
                if k2 == 0:
                    data1Pert.pop(k)
                    data1Pert.insert(k, list(np.zeros(len(data1[0]))))
                elif k2 == 1:
                    data2Pert.pop(k)
                    data2Pert.insert(k, list(np.zeros(len(data2[0]))))
                elif k2 == 2:
                    data3Pert.pop(k)
                    data3Pert.insert(k, list(np.zeros(len(data3[0]))))
                elif k2 == 3:
                    data4Pert.pop(k)
                    data4Pert.insert(k, list(np.zeros(len(data4[0]))))
                elif k2 == 4:
                    data5Pert.pop(k)
                    data5Pert.insert(k, list(np.zeros(len(data5[0]))))
                elif k2 == 5:
                    data6Pert.pop(k)
                    data6Pert.insert(k, list(np.zeros(len(data6[0]))))
    else:
        if type != 'none':
            print("type not specified")

    return data1Pert, data2Pert, data3Pert, data4Pert, data5Pert, data6Pert




#For numerical features
#----------------------
#For type = 'gaussian': level is standard deviation of the normally distributed noise. 
#For type = 'zero': level is the number of features to be masked.

def addPerturbationToNum(type, level, data):
    dataNPert = copy.deepcopy(data)

    if type == 'gaussian':                                              
        for k in range(len(data)):                                             #gaussian noise affects all samples
            for k2 in range(len(dataNPert[0])): #for each numerical feature
                dataNPert[k][k2] += np.random.normal(loc = 0.0, scale = level)  #here 'level' means standard deviation of normally distributed noise
    elif type == 'zero':
        for k in range(len(data)):                                             #zero noise affects all samples, but not all features for each sample
            picked = randomIndices(level, 3)                                   #chose 'level' number of features to be affected by noise for this sample
            for k2 in picked:
                dataNPert[k][k2] = 0.0
    else:
        if type != 'none':
            print("type not specified")

    return(dataNPert)
