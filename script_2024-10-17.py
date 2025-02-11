import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import csv
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import os
import math
import sys
import random
import copy

from AE_noise import *
from data_handeling import *
from measuring_functions import *

#This makes matplotlib work
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Using samples 0-80000 as training set, 80000-100000 as test set.
#and create plot, and save result data to be able to create concentration curves

#corrected, so that AE works


epochsCatAE = 15
epochsNumAE = 15
epochsFF = 198
lrCatAE = 0.005
lrNumAE = 0.005
lrFF = 0.0001
usingDenoising = True


numNoiseType = 'gaussian'  #only used if usingDenoising is true
numNoiseLevel = 0.1


catNoiseType = 'sample'  #only used if usingDenoising is true
catNoiseLevel = 2

useDropout = False
dropoutLevel = 0.2


cutOffLevel = 54 #we do not remove anything

nSamples = 80000          #60000
nSamplesTest = 20000

#4,6,8,10,15,25,50,80

nCatLayer = 8

#nLayers =  int(sys.argv[1])        #1; number of hidden layers
#nHiddenLayer = int(sys.argv[2])    #6; number of neurons in each hidden layer

nLayers = 3
nHiddenLayer1 = int(sys.argv[1])
nHiddenLayer2 = int(sys.argv[2])
nHiddenLayer3 = int(sys.argv[3])


if usingDenoising == True:
    outputFileName = "outputWithDenoisedAE" + str(nHiddenLayer1) + "-" + str(nHiddenLayer2) + "-" + str(nHiddenLayer3) + ".txt"
else:
    outputFileName = "outputWithAE" + str(nHiddenLayer1) + "-" + str(nHiddenLayer2) + "-" + str(nHiddenLayer3) + ".txt"
    numNoiseType = 'none'
    catNoiseType = 'none'


answerPDL = np.zeros(4)
answerPDLall = np.zeros(20)

print("\n\nProgram started...\n")

print("Importing data...")

allDataCat = import_data('..\dataKat2.csv', 'categorical')
allDataNum = import_data('..\dataNumeric2.csv', 'numerical')
allDataExp = import_data('..\dataExposure2.csv', 'exposure')
allDataCla = import_data('..\dataClaims2.csv', 'claims')

for jj in range(1):           #5
    for ii in range(1):       #4

        print("Picking training data and test data...\n")

        #chooseSet = ii + 1 

        #testStart = 60000    #first index of test set
        #testEnd = 60000          #last index of test set
        #validationEnd = 80000
        #dataEnd = 100000         #total number of samples


        #Validation set
        #dataCatVal = allDataCat[60000:80000]
        #dataNumVal = allDataNum[60000:80000]
        #dataExposureVal = allDataExp[60000:80000]
        #dataClaimsVal = allDataCla[60000:80000]

        #Test set
        dataCatTest = allDataCat[80000:100000]
        dataNumTest = allDataNum[80000:100000]
        dataExposureTest = allDataExp[80000:100000]
        dataClaimsTest = allDataCla[80000:100000]


        #Training set
        dataCat = allDataCat[0:80000]
        dataNum = allDataNum[0:80000]
        dataExposure = allDataExp[0:80000]
        dataClaims = allDataCla[0:80000]

        #Categorical features (training set) for training categorical autoencoder
        dataCat1 = get_one_category(dataCat, 1)
        dataCat2 = get_one_category(dataCat, 2)
        dataCat3 = get_one_category(dataCat, 3)
        dataCat4 = get_one_category(dataCat, 4)
        dataCat5 = get_one_category(dataCat, 5)
        dataCat6 = get_one_category(dataCat, 6)
        

        #Possibly add noise for autoencoder training
        #-------------------------------------------------------------------------------
        
        #Categorical features
        print("Adding noise to categorical features input for AE...")
        print("Noise type: " + catNoiseType + "\n")
        dataCat1pert, dataCat2pert, dataCat3pert, dataCat4pert, dataCat5pert, dataCat6pert = addPerturbationToCat(catNoiseType, catNoiseLevel, dataCat1, dataCat2, dataCat3, dataCat4, dataCat5, dataCat6)

        #Numerical features
        print("Adding noise to numerical features input for AE...")
        print("Noise type: " + numNoiseType + "\n")
        dataNumPert = addPerturbationToNum(numNoiseType, numNoiseLevel, dataNum)


        #AUTO-ENCODER FOR CATEGORIES
        #-------------------------------------------------------------------------------
        print("Creating Auto-encoder model")
        print("")

        input1 = keras.Input(shape=(6,))
        input2 = keras.Input(shape=(3,))
        input3 = keras.Input(shape=(7,))
        input4 = keras.Input(shape=(11,))
        input5 = keras.Input(shape=(6,))
        input6 = keras.Input(shape=(21,))

        xAE = tf.keras.layers.Concatenate()([input1, input2, input3, input4, input5, input6])

        hiddenAE = layers.Dense(8, activation=linearAct, use_bias=False, name='hiddenAE', kernel_initializer=initializers.GlorotUniform())(xAE)

        output1 = layers.Dense(6, activation='softmax', name='out1', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE)   #vehPower
        output2 = layers.Dense(3, activation='softmax', name='out2', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE)   #vehAge
        output3 = layers.Dense(7, activation='softmax', name='out3', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE)   #drivAge
        output4 = layers.Dense(11, activation='softmax', name='out4', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE)  #vehBrand
        output5 = layers.Dense(6, activation='softmax', name='out5', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE)   #Area
        output6 = layers.Dense(21, activation='softmax', name='out6', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE)  #Region

        modelAE = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=[output1, output2, output3, output4, output5, output6])

        modelAE.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=lrCatAE), loss=tf.keras.losses.CategoricalCrossentropy(), loss_weights=[1., 1., 1., 1., 1., 1.])

        print("Training Categorical AE...") #possibly use perturbed input data
        modelAE.fit(x=[np.array(dataCat1pert), np.array(dataCat2pert), np.array(dataCat3pert), np.array(dataCat4pert), np.array(dataCat5pert), np.array(dataCat6pert)], y=[np.array(dataCat1), np.array(dataCat2), np.array(dataCat3), np.array(dataCat4), np.array(dataCat5), np.array(dataCat6)], batch_size=1000, epochs=epochsCatAE, validation_split=0.0)

        weights_AE1 = modelAE.get_layer(name='hiddenAE').get_weights()


        #CREATE NUMERICAL AE INPUT/OUTPUT
        #-------------------------------------------------------------------------------
        #Network mimicing the categorical AE encoder-part. Use to obtain numerical input/output
        #We don't train this network, only use the weights from categorical AE and then predict.

        input1m = keras.Input(shape=(6,))
        input2m = keras.Input(shape=(3,))
        input3m = keras.Input(shape=(7,))
        input4m = keras.Input(shape=(11,))
        input5m = keras.Input(shape=(6,))
        input6m = keras.Input(shape=(21,))

        xAEm = tf.keras.layers.Concatenate()([input1m, input2m, input3m, input4m, input5m, input6m])

        output_m = layers.Dense(8, activation=linearAct, use_bias=False, name='output_m')(xAEm)

        modelAEm = keras.models.Model(inputs=[input1m, input2m, input3m, input4m, input5m, input6m], outputs=output_m)

        modelAEm.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.01), loss=tf.keras.losses.CategoricalCrossentropy()) #learning rate etc does not matter since do not train this network

        #Set weights of this model to same as categorical AE
        encoder_weights = modelAE.get_layer(name='hiddenAE').get_weights()


        #Set weights
        modelAEm.get_layer(name='output_m').set_weights(encoder_weights)
        sampsAEm = [np.array(dataCat1), np.array(dataCat2), np.array(dataCat3), np.array(dataCat4), np.array(dataCat5), np.array(dataCat6)]
        resAEm = modelAEm.predict(sampsAEm)

        #Rescale weights
        rescaled_encoder_weights = encoder_weights
        for j in range(8):
            maxval = max(resAEm[:,j])
            minval = min(resAEm[:,j])
            for i in range(54):
                rescaled_encoder_weights[0][i][j] = rescaled_encoder_weights[0][i][j]*2/(maxval-minval) - 2*minval/(6*(maxval-minval))+1/6 #correction: +1/6 instead of -1/6

        #Set weights to rescaled encoder weights
        modelAEm.get_layer(name='output_m').set_weights(rescaled_encoder_weights)


        print("Creating representations of categorical variables based on AE1 ...")
        resAEm = modelAEm.predict(sampsAEm)


        #Concatenate the representations for the categories with numerical input
        n = len(resAEm)
        numAESamples = []
        numAESamplesPert = []

        print("Concatenating representations of categories with numerical input...")
        for i in range(n):
            newElement = list(resAEm[i])       #Without perturbation
            newElement.append(dataNum[i][0])
            newElement.append(dataNum[i][1])
            newElement.append(dataNum[i][2])
            numAESamples.append(newElement)
            newElement = list(resAEm[i])       #With perturbation
            newElement.append(dataNumPert[i][0])
            newElement.append(dataNumPert[i][1])
            newElement.append(dataNumPert[i][2])
            numAESamplesPert.append(newElement)

        #AUTO-ENCODER NUMERICAL
        #-------------------------------------------------------------------------------
        print("\n\nCreating numerical AE...")

        inputsAE2 = keras.Input(shape=(11,))
        hiddenAE2 = layers.Dense(nHiddenLayer1, activation="tanh", name='hiddenAE2', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(inputsAE2)      #nHiddenLayer1 now!
        outputsAE2 = layers.Dense(11, activation=linearAct, kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.GlorotUniform())(hiddenAE2)

        modelAE2 = keras.Model(inputsAE2, outputsAE2)

        modelAE2.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=lrNumAE), loss=tf.keras.losses.MeanSquaredError())

        print("Fitting numerical AE...")
        modelAE2.fit(x=np.array(numAESamplesPert), y=np.array(numAESamples), batch_size=1000, epochs=epochsNumAE, validation_split=0.0) #use samples with perturbed numerical features as input

        weights_AE2 = modelAE2.get_layer(name='hiddenAE2').get_weights()



        #FEED-FORWARD NETWORK FOR PREDICTING N.O. CLAIMS 
        #--------------------------------------------------------------------------------------


        #callback for early stopping condition
        #callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, min_delta = 0, mode = 'min', restore_best_weights=True)         #do not use callback this time

        inputsCat = keras.Input(shape=(54,))
        outputCat = layers.Dense(nCatLayer, activation=linearAct, use_bias=False, name='catHidden', kernel_initializer=initializers.GlorotUniform())(inputsCat) #use Xavier (=GlorotUniform()) initializer
        inputsNum = keras.Input(shape=(3,))
        x = tf.keras.layers.Concatenate()([outputCat, inputsNum])

        #use only three layers this time, with different numbers of neurons
        if useDropout == True:
            x = layers.Dropout(dropoutLevel)(x)
        x = layers.Dense(nHiddenLayer1, activation="tanh", name='hidden1', kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())(x)
        if (nLayers > 1):
            if useDropout == True:
                x = layers.Dropout(dropoutLevel)(x)
            x = layers.Dense(nHiddenLayer2, activation="tanh", kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())(x)
        if (nLayers > 2):
            if useDropout == True:
                x = layers.Dropout(dropoutLevel)(x)
            x = layers.Dense(nHiddenLayer3, activation="tanh", kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros())(x)
        if useDropout == True:
            x = layers.Dropout(dropoutLevel)(x)
        x = layers.Dense(1, activation="exponential")(x)

        inputExp = keras.Input(shape=(1,))
        output = tf.keras.layers.Multiply()([x, inputExp])

        model = keras.Model([inputsCat, inputsNum, inputExp], output)
        model.summary()


        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=lrFF), loss=poisson_deviance_loss) #use learning rate=1*10^(-4) this time


        #Set weights for >>repressenting the categorical<<-layer to AE1 weights
        model.get_layer(name='catHidden').set_weights(rescaled_encoder_weights)  #correction: rescaled_encoder_weights instead of weights_AE1


        #Set weights for hidden layer to AE2 weights
        model.get_layer(name='hidden1').set_weights(weights_AE2)



        #validation set input
        #sampsVal = [np.array(dataCatVal), np.array(dataNumVal), np.array(dataExposureVal)]


        print("Fit the model to the data")
        # Fit the model
        #do not use callback this time
        #history = model.fit(x=[np.array(dataCat), np.array(dataNum), np.array(dataExposure)], y=np.array(dataClaims), batch_size=1000, epochs=1000, callbacks=[callback], validation_data=(sampsVal,np.array(dataClaimsVal))) 
        history = model.fit(x=[np.array(dataCat), np.array(dataNum), np.array(dataExposure)], y=np.array(dataClaims), batch_size=1000, epochs=epochsFF, validation_split=0.0) 

        print("epochs trained: " + str(len(history.history['loss'])))

        outputFile = open(outputFileName,"a")
        outputFile.write("Number of epochs trained for run " + str(jj) + ", set "+str(ii)+" is: " + str(len(history.history['loss']))+"\n")

        #Save model-------------------------------------------------------------------------------------------------------

        #modelName = "modelWithDenoisedAE" + str(nHiddenLayer1) + "-" + str(nHiddenLayer2) + "-" + str(nHiddenLayer3) + "_Set" + str(ii) + "_Run" + str(jj) + ".keras"
        modelName ='modelAE.keras'
        model.save(modelName)

        #-----------------------------------------------------------------------------------------------------------------

        #test set input
        sampsTest = [np.array(dataCatTest), np.array(dataNumTest), np.array(dataExposureTest)]

        
        res = model.predict(sampsTest)

        print("called the model:")
        print(res)
        print(res[0])
        print("sum predicted claims:")
        print(np.sum(res))
        print("")
        print("this is: ")
        print(np.array(dataExposureTest[0:2]))

        sumExp = np.sum(dataExposureTest)
        sumClaims = np.sum(dataClaimsTest)
        percentageClaims = sumClaims/sumExp

        percentageClaimsPredicted = np.sum(res)/sumExp

        print("")
        print("Sum Exposure: ")
        print(sumExp)
        print("")
        print("Sum Claims: ")
        print(sumClaims)
        print("")
        print("Percentage claims: ")
        print(percentageClaims)
        print("")
        print("Precentage predicted claims: ")
        print(percentageClaimsPredicted)


        print("------------------")

        rRes = np.reshape(res,len(res))

        riskRes = np.zeros(len(rRes))
        exposures = np.array(dataExposureTest)

        for i in range(len(rRes)):
           riskRes[i] = rRes[i]/exposures[i]


        indexes = np.argsort(riskRes)
        print("The res:")
        print(rRes)
        print("sorted they are:")
        print(indexes)

        sortedRes = np.zeros(len(rRes))

        theClaims = np.array(dataClaimsTest)
        sortedClaims = np.zeros(len(rRes))

        for i in range(len(rRes)):
            sortedRes[i] = rRes[indexes[i]]
            sortedClaims[i] = theClaims[indexes[i]]


        #plt.plot(sortedRes)
        #plt.plot(sortedClaims, color='r')
        #plt.show()

        stN=250
        diffSt = int(math.floor(len(rRes)/stN))

        stClaims = np.zeros(stN)
        stRes = np.zeros(stN)

        j = 0
        k = 0
        for i in range(len(rRes)):
            stRes[j] += sortedRes[i]
            stClaims[j] += sortedClaims[i]
            k += 1
            if (k == diffSt):
                stRes[j] = stRes[j]/diffSt
                stClaims[j] = stClaims[j]/diffSt
                j += 1
                k = 0

        print("length rRes:")
        print(len(rRes))
        print("length diff:")
        print(diffSt)

        pdlValue = poisson_deviance_loss(np.array(theClaims[:,0],dtype="float32"), np.array(rRes,dtype="float32"))
        print("Poisson deviance loss on test set: ")
        print(float(pdlValue))
        answerPDL[ii] = float(pdlValue)
        answerPDLall[jj*4 + ii] = float(pdlValue)
        print(ii)
        print(jj)

    print(answerPDL)
    print("mean: ")
    print(np.mean(answerPDL))


    #To copy easily to latex
    outputFile = open(outputFileName,"a")
    outputFile.write("Run ")
    outputFile.write(str(jj))
    outputFile.write(": ")
    
    for i in range(4):
        outputFile.write(str(answerPDL[i].round(decimals=8)))
        outputFile.write(" & ")

    mn = np.mean(answerPDL)
    outputFile.write(str(mn.round(decimals=7)))
    outputFile.write("\n")
    outputFile.close()

mn = np.mean(answerPDLall)
vr = np.sqrt(np.var(answerPDLall))

outputFile = open(outputFileName,"a")
outputFile.write("total mean: ")
outputFile.write(str(mn))
outputFile.write("\ntotal st-dev: ")
outputFile.write(str(vr))
outputFile.write("\n")
outputFile.close()
                


                
#Save true claims and predicted claims



theFileName = "y_pred.txt"
theFileName2 = "y_true.txt"


theFile = open(theFileName,"a")
theFile2 = open(theFileName2,"a")

for i in range(len(rRes)):
    theFile.write(str(rRes[i]))
    theFile.write("\n")
    theFile2.write(str(theClaims[i][0]))
    theFile2.write("\n")

theFile.close()
theFile2.close()


#thePlotTitle = "Risk-ordering predictions, " + str(nHiddenLayer1) + ", no AE"
#thePlotTitle = "Risk-ordering predictions, " + str(nHiddenLayer1) + "-" + str(nHiddenLayer2) + "-" + str(nHiddenLayer3) + ", no AE"

if usingDenoising == True:
    thePlotTitle = "Risk-ordering predictions, " + str(nHiddenLayer1) + "-" + str(nHiddenLayer2) + "-" + str(nHiddenLayer3) + ", denoising AE"
else:
    thePlotTitle = "Risk-ordering predictions, " + str(nHiddenLayer1) + "-" + str(nHiddenLayer2) + "-" + str(nHiddenLayer3) + ", with AE"

plt.plot(stClaims, 'ro')
plt.plot(stRes,'b')
plt.legend(["Empirical", "Predicted"])
plt.title(thePlotTitle)
plt.xlabel('Probability bins')
plt.ylabel('No. claims')
plt.show()
