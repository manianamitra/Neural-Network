from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import math
######### 1 For SPAM 0 for HAM ########
################## Preprocessing and Vectoriaing Training and Test Data Set##############
preProcessedDictTrain = {}
preProcessedDictTest = {}
vectorDictTrain = {} #For neural network
vectorDictTest = {}  #For neural network
uniqueWords = set()
DEF_TRAINING_LIMIT = 4460
DEF_THRESHOLD = 0.7
#######################################################################################

################## Parameters for neural network #####################################
neuralNetDescription = [-1,100,50,1] #Only nmber of neurons no bias
TOTAL_LAYERS = len(neuralNetDescription)
BIAS = 1
deltaDict = {}
weightMatrixDict = {}
learnedFeaturesDict = {}
LEARNING_RATE = 0.1
NUMBER_OF_EPOC = 10
#######################################################################################

######################## pyplot ######################################################

xAxis = []
yAxisForInSample = []
yAxisForOutSample = []
######################################################################################

####################### Neural Network Implementation #################################
def tanH(x):
     return np.tanh(x)
      
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def derivativeSigmoid(x):
    return (x * (1-x))   #x = sigmoid(x)

def derivativeTanH(x):
    return (1 - (x*x))   #x=tanh(s)

def updateWeight(layer):
    global weightMatrixDict
    global learnedFeaturesDict
    global deltaDict
    weightMatrix = weightMatrixDict.get(layer)
    featureMatrix = learnedFeaturesDict.get(layer-1)
    deltaMatrix = deltaDict.get(layer)
    noOfRowsInDeltaMatrix = deltaMatrix.shape[0]
    gradientList = []
    for row in range(1,noOfRowsInDeltaMatrix):
        deltaValue = deltaMatrix[row,0] 
        featureMatrixNew = ((deltaValue*LEARNING_RATE)*featureMatrix)
        featureMatrixT = featureMatrixNew.transpose() 
        featureMatrixList = featureMatrixT.tolist() 
        gradientList.append(featureMatrixList[0])
    gradientVector = np.array(gradientList)
    weightMatrixUpdated = weightMatrix - gradientVector
    weightMatrixDict.update({layer:weightMatrixUpdated})



def forwardPass(inputVector):
    global weightMatrixDict
    global learnedFeaturesDict
    learnedFeaturesDict ={}  #Reinitializing feature vector
    learnedFeaturesDict.update({0:inputVector})
    arrayOfOne = np.array([[BIAS]])
    for layer in range (1,TOTAL_LAYERS):
        weightMatrix = weightMatrixDict.get(layer)
        inputVector = learnedFeaturesDict.get(layer-1)
        outVector = np.matmul(weightMatrix,inputVector)
        outVector = tanH(outVector)
        outVector = np.concatenate((arrayOfOne, outVector), axis=0)
        learnedFeaturesDict.update({layer:outVector})    
        
def backwardPass(correctPrediction):
    global deltaDict
    deltaDict = {} #Reinitializing delta vector
    lastLayerValue = learnedFeaturesDict.get(TOTAL_LAYERS-1)  
    predictedValue = lastLayerValue[1,0]
    errorInPrediction = predictedValue - correctPrediction
    deltaLastLayer = errorInPrediction * derivativeTanH(predictedValue)
    deltaVector = np.array([[0],[deltaLastLayer]])
    deltaDict.update({(TOTAL_LAYERS-1):deltaVector})
    for layer in range(TOTAL_LAYERS-2,0,-1):
        weightMatrix = weightMatrixDict.get(layer+1)    #Get the weight Vector of next layer
        weightMatrix = weightMatrix.transpose() 
        numOfRows = weightMatrix.shape[0]
        zeroArray = np.zeros(numOfRows)
        weightMatrix = np.hstack((zeroArray[:, np.newaxis],weightMatrix))  #Adding a column of zeros
        deltaVectorOfNextLayer = deltaDict.get(layer+1)                    #Getting the delta vector of next layer
        deltaStarCurrentLayer = np.matmul(weightMatrix,deltaVectorOfNextLayer)  #DelataStar For current layer 
        currentFeature = learnedFeaturesDict.get(layer)  #Getting current feature vector 
        currentFeature = derivativeTanH(currentFeature)
        currentFeature = currentFeature.transpose()
        intermediateVector = np.diag(currentFeature[0])
        deltaCurrentLayer = np.matmul(intermediateVector,deltaStarCurrentLayer)   #Multiplying with derivative of activator 
        deltaDict.update({layer:deltaCurrentLayer})
        
def createNN():
    global weightMatrix
    
    for layer in range(1,TOTAL_LAYERS):
        weightMatrix = np.random.randint( 10, size = (neuralNetDescription[layer],(neuralNetDescription[layer-1]+1)))
        weightMatrixDict.update({layer:weightMatrix})

#######################################################################################

########################## Vectorizing Training Data and Test Data ####################

def createVectors():  #This modules Creates Vectors
    global preProcessedDictTrain
    global preProcessedDictTest
    global vectorDictTrain
    global vectorDictTest

    listOfSpams = preProcessedDictTrain.get('spam')
    listOfHams = preProcessedDictTrain.get('ham')

    listOfSpamsTest = preProcessedDictTest.get('spam') #Added 
    listOfHamsTest = preProcessedDictTest.get('ham') #Added
    
    listOfSpamVectors = []
    listOfHamVectors = []
    #########################################################
    for listOfWords in listOfSpams:
        vectorForm = []
        for word in uniqueWords:
            if word in listOfWords:
               vectorForm.append(1)
            else:
               vectorForm.append(0)
        listOfSpamVectors.append(vectorForm)

    for listOfWords in listOfHams:
        vectorForm = []
        for word in uniqueWords:
            if word in listOfWords:
               vectorForm.append(1)
            else:
               vectorForm.append(0)
        
        listOfHamVectors.append(vectorForm)
    vectorDictTrain.update({'spam':listOfSpamVectors})
    vectorDictTrain.update({'ham':listOfHamVectors})
    print 'Training data vectorized.'
    ###############################################################
    listOfSpamVectors = []
    listOfHamVectors = []
    for listOfWords in listOfSpamsTest:
        vectorForm = []
        for word in uniqueWords:
            if word in listOfWords:
               vectorForm.append(1)
            else:
               vectorForm.append(0)
        listOfSpamVectors.append(vectorForm)
   ##################################################################
    for listOfWords in listOfHamsTest:
        vectorForm = []
        for word in uniqueWords:
            if word in listOfWords:
               vectorForm.append(1)
            else:
               vectorForm.append(0)
        
        listOfHamVectors.append(vectorForm)
   ###################################################################
    vectorDictTest.update({'spam':listOfSpamVectors})
    vectorDictTest.update({'ham':listOfHamVectors})
   ###################################################################
    print 'Test data vectorized.'
    spamList = vectorDictTrain.get('spam')


def preProcessData():
    global preProcessedDictTrain
    global preProcessedDictTest
    global uniqueWords
    file = open ('data.txt','r')
    iterator = 0
    stop_words = set(stopwords.words('english'))   #Stop Words
    stemmer = PorterStemmer()   #Stemmer For pre processing
    processedListSpam = []
    processedListHam = []
    processedListSpamTest = []
    processedListHamTest = []
    for line in file:
         
         if (iterator < DEF_TRAINING_LIMIT):
		fileLine = line.split('\t')
		if ( fileLine[0] == 'spam' or fileLine[0] == 'ham' ):
		    listOfWords = fileLine[1].split(' ')
		    listOfWords = [ re.sub('[!@#$\n.]', '', word) for word in listOfWords]  #removing special characters
		    listOfWords = [ word.lower() for word in listOfWords ]  #Case unfolding
		    listOfWords = [ re.sub(r'[^\x00-\x7F]+',' ',word) for word in listOfWords] #removing non-ASCII characters
		    listOfWords = [stemmer.stem(word) for word in listOfWords]   #Stemming
		    listOfWords = [word for word in listOfWords if not word in stop_words]   #Removing stop words
		    for word in listOfWords:
		      uniqueWords.add(word)
		    if (fileLine[0] == 'spam'):
		       processedListSpam.append(listOfWords)
		    else:
		       
		       processedListHam.append(listOfWords)
		iterator = iterator +1
         else:
               fileLine = line.split('\t')
	       if ( fileLine[0] == 'spam' or fileLine[0] == 'ham' ):
		    listOfWords = fileLine[1].split(' ')
		    listOfWords = [ re.sub('[!@#$\n.]', '', word) for word in listOfWords]  #removing special characters
		    listOfWords = [ word.lower() for word in listOfWords ]  #Case unfolding
		    listOfWords = [ re.sub(r'[^\x00-\x7F]+',' ',word) for word in listOfWords] #removing non-ASCII characters
		    listOfWords = [stemmer.stem(word) for word in listOfWords]   #Stemming
		    listOfWords = [word for word in listOfWords if not word in stop_words]   #Removing stop words
		    if (fileLine[0] == 'spam'):
		       processedListSpamTest.append(listOfWords)
		    else:
		       processedListHamTest.append(listOfWords)
	       iterator = iterator +1
                  
    file.close()
    preProcessedDictTrain.update({'spam':processedListSpam})
    preProcessedDictTrain.update({'ham':processedListHam})
    preProcessedDictTest.update({'spam':processedListSpamTest})
    preProcessedDictTest.update({'ham':processedListHamTest})
    
######################################################################################################

def inSampleError():
    global vectorDictTrain
    listOfSpamVectors = vectorDictTrain.get('spam')
    listOfHamVectors = vectorDictTrain.get('ham')
    totalSize = len(listOfSpamVectors) + len(listOfHamVectors)
    error = 0
    for listOfVectorValues in listOfSpamVectors: 
        inputVectorList = [BIAS] + listOfVectorValues
	inputVector = np.array([inputVectorList])
	inputVector = inputVector.transpose()
	forwardPass(inputVector)
        vectorAtLastLayer = (learnedFeaturesDict.get(TOTAL_LAYERS-1))
        predictedValue = vectorAtLastLayer[1,0]
        if (predictedValue >= DEF_THRESHOLD):
            error = error + 0
        else:
            error = error + 1

    for listOfVectorValues in listOfHamVectors: 
        inputVectorList = [BIAS] + listOfVectorValues
	inputVector = np.array([inputVectorList])
	inputVector = inputVector.transpose()
	forwardPass(inputVector)
        vectorAtLastLayer = (learnedFeaturesDict.get(TOTAL_LAYERS-1))
        predictedValue = vectorAtLastLayer[1,0]
        if (predictedValue >= DEF_THRESHOLD):
           error = error + 1
        else:
           error = error + 0
    return float(error)/float(totalSize)
    
def outSampleError():
    global vectorDictTest
    listOfSpamVectors = vectorDictTest.get('spam')
    listOfHamVectors = vectorDictTest.get('ham')
    totalSize = len(listOfSpamVectors) + len(listOfHamVectors)
    error = 0
    for listOfVectorValues in listOfSpamVectors: 
        inputVectorList = [BIAS] + listOfVectorValues
	inputVector = np.array([inputVectorList])
	inputVector = inputVector.transpose()
	forwardPass(inputVector)
        vectorAtLastLayer = (learnedFeaturesDict.get(TOTAL_LAYERS-1))
        predictedValue = vectorAtLastLayer[1,0]
        if (predictedValue >= DEF_THRESHOLD):
            error = error + 0
        else:
            error = error + 1

    for listOfVectorValues in listOfHamVectors: 
        inputVectorList = [BIAS] + listOfVectorValues
	inputVector = np.array([inputVectorList])
	inputVector = inputVector.transpose()
	forwardPass(inputVector)
        vectorAtLastLayer = (learnedFeaturesDict.get(TOTAL_LAYERS-1))
        predictedValue = vectorAtLastLayer[1,0]
        if (predictedValue >= DEF_THRESHOLD):
           error = error + 1
        else:
           error = error + 0
    return float(error)/float(totalSize)

def trainNeuralNetwork():
    global neuralNetDescription
    global vectorDictTrain
    global neuralNetDescription
    global uniqueWords
    global XAxis
    global yAxisForInSample
    global yAxisForOutSample 
    #inputList = [[BIAS,0,0,1]]
    neuralNetDescription[0] = len(uniqueWords)  #Setting the number of neurons for first layer
    listOfSpamVectors = vectorDictTrain.get('spam')
    listOfHamVectors = vectorDictTrain.get('ham')
    createNN()   #Creating the initial neural network
    for noOfEpoc in range(0,NUMBER_OF_EPOC):
	    ##### Train For Spam Dataset ##################
            print 'Iteration No:'+str(noOfEpoc)
	    for listOfVectorValues in listOfSpamVectors: 
		inputVectorList = [BIAS] + listOfVectorValues
		inputVector = np.array([inputVectorList])
		inputVector = inputVector.transpose()
		forwardPass(inputVector)
		backwardPass(1)
		for layer in range(1,TOTAL_LAYERS):
		    updateWeight(layer)
	    print 'Training For SPAM Dataset completed.'
	    ##############################################
	    ##### Train for HAM dataset ##################
	    for listOfVectorValues in listOfHamVectors: 
		inputVectorList = [BIAS] + listOfVectorValues
		inputVector = np.array([inputVectorList])
		inputVector = inputVector.transpose()
		forwardPass(inputVector)
		backwardPass(0)
		for layer in range(1,TOTAL_LAYERS):
		    updateWeight(layer)
	    print 'Training For HAM Dataset completed.'
            inError = inSampleError()
            outError = outSampleError()
            xAxis.append(noOfEpoc)
            print inError
            print outError
            yAxisForInSample.append(inError)
            yAxisForOutSample.append(outError)
	    ##############################################
   

def scheduler():
  preProcessData()
  print 'Data Preprocessed.'
  createVectors()
  print 'Vectors Created.'
  trainNeuralNetwork()
  print 'Traning Completed'
  plt.plot(xAxis,yAxisForInSample)
  plt.savefig('InSample.png', bbox_inches='tight')
  plt.plot(xAxis,yAxisForOutSample)
  plt.savefig('outSample.png', bbox_inches='tight')
  

scheduler()  
    
