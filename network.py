# created 24 Sep 2020, Teddy Drewes
# last update 29 Oct 2020
###############################################
# The network object
# Runs the network
###############################################

import numpy as np
import random
import math
from NMISTdata import MNISTdataDriver as MNISTdd

class Network:

    def __init__(self):
        self.N = 784
        self.M = 10
        self.D = 3
        self.Nodes = [784,100,32,16,10]
        self.biasRange = [-10,10]
        self.weightRange = [-1,1]
        self.numTests = 10000
        self.numTraining = 60000
        self.testInput=[]
        self.trainInput=[]
        self.correctOutput = 0
        self.output= []
        self.backprop = True
        self.stochaticGroup = 100
        # numbers for analysis
        self.totalTests = 0
        self.totalRight = 0

    def __str__(self):
        return  ('\nInputs: ' + str(self.N) + '\n' \
        'Outputs: ' + str(self.M) + '\n' \
        'Hidden Layers: ' + str(self.D) + '\n' \
        'Width of Layers Input-->Output ' + str(self.Nodes) + '\n' \
        'Bias Range: ' + str(self.biasRange[0]) + ' to ' + str(self.biasRange[1]) + '\n' \
        'Weight Range: ' + str(self.weightRange[0]) + ' to ' + str(self.weightRange[1]))

    def setNetworkSize(self):
        #self.N = int(input("How many inputs are there? "))
        #self.Nodes.append(self.N)
        #self.M = int(input("How many possible outputs are there? "))
        #self.D = int(input("How many hidden layers would you like? "))
        #for i in range (self.D):
        #    self.Nodes.append(int(input("How many nodes in Layer " + str(i) + "? ")))
        #self.Nodes.append(self.M)
        #self.biasRange[0] = float(input("Minimum Bias value? "))
        #self.biasRange[1] = float(input("Maximum Bias value? "))
        #self.weightRange[0] = float(input("Minimum Weight value? "))
        #self.weightRange[1] = float(input("Maximum weight value? "))
        #self.numTraining = int(input("number of Training Data: "))
        #self.numTests = int(input("Number of Training Tests: "))
        #networkType = int(input("Back-propagation (0) or only Forward-propagation (1)? "))
        #if networkType == 0:
        #    self.backprop = True
        #else:
        #    self.backprop = False
        self.setInputMNISTtesting(self.numTests)
        self.setInputMNISTtraining(self.numTraining)
        self.setOutput(self.M)

    #reads in the number of test sets to be used
    def setInputMNISTtesting(self, numInputSets):
        # implement code here for this/can change based on data set
        print("setting NMIST input up")
        myMNIST = MNISTdd()
        myMNIST.runDriver()
        for i in range(numInputSets):
            self.testInput.append((myMNIST.data_dict['test images'][i],myMNIST.data_dict['test labels'][i]))

    #reads in the number of training sets to be used
    def setInputMNISTtraining(self, numInputSets):
        # implement code here for this/can change based on data set
        print("setting NMIST input up")
        myMNIST = MNISTdd()
        myMNIST.runDriver()
        for i in range(numInputSets):
            self.trainInput.append((myMNIST.data_dict['train images'][i],myMNIST.data_dict['train labels'][i]))



    # sets the output list for what each node will correspond to
    def setOutput(self, numOutputs):
        for i in range(numOutputs):
            self.output.append(int(input("What does the " + str(i) + " output correspond to? ")))

    def initMatrix(self):
        #define weight matrixes
        self.weightMatrix = []
        for i in range (1,len(self.Nodes)):
            self.weightMatrix.append(np.empty([self.Nodes[i-1],self.Nodes[i]],dtype=float))
        for mat in self.weightMatrix:
            for i in range (0,len(mat)):
                for j in range(0,len(mat[i])):
                    mat[i][j] = random.uniform(self.weightRange[0],self.weightRange[1])
        #print(self.weightMatrix)

        #define bias matrixes
        self.biasMatrix = []
        for i in range(1, len(self.Nodes)):
            self.biasMatrix.append(np.empty([1, self.Nodes[i]], dtype=float))
        for mat in self.biasMatrix:
            for i in range(0, len(mat)):
                mat[i] = random.uniform(self.biasRange[0], self.biasRange[1])
        #print(self.biasMatrix)

        #define Node Matrixes
        self.nodeMatrix = []
        for i in range(len(self.Nodes)):
            self.nodeMatrix.append(np.empty([1,self.Nodes[i]],dtype = float))
        #print(self.nodeMatrix)
        
    def initDeltaMatrixes(self):
        # define deltaWeight matrixes
        self.deltaWeightMatrix = []
        for i in range(1, len(self.Nodes)):
            self.deltaWeightMatrix.append(np.empty([self.Nodes[i - 1], self.Nodes[i]], dtype=float))
        for mat in self.deltaWeightMatrix:
            for i in range(0, len(mat)):
                for j in range(0, len(mat[i])):
                    mat[i][j] = random.uniform(0, 0)
        #print(self.deltaWeightMatrix)
        
        # define deltaCost matrixes
        self.deltaCostMatrix = []
        for i in range(1, len(self.Nodes)):
            self.deltaCostMatrix.append(np.empty([1, self.Nodes[i]], dtype=float))
        for mat in self.deltaCostMatrix:
            for i in range(0, len(mat)):
                mat[i] = random.uniform(0,0)
        # print(self.deltaCostMatrix)

        # define deltaBias matrixes
        self.deltaBiasMatrix = []
        for i in range(1, len(self.Nodes)):
            self.deltaBiasMatrix.append(np.empty([1, self.Nodes[i]], dtype=float))
        for mat in self.deltaBiasMatrix:
            for i in range(0, len(mat)):
                mat[i] = random.uniform(0,0)
        # print(self.deltaBiasMatrix)

        self.answerMatrix = []
        for i in range(10):
            self.answerMatrix.append(0)

    def setCorrect(self, correct):
        for i in range(10):
            if i == correct:
                self.answerMatrix[i] = 1
            else:
                self.answerMatrix[i] = 0
    
    # runs the network
    def runNetwork(self):

        # training on backprop
        for i in range(0,self.numTraining,100):
            #initialize all the matrixes for use in stochic groupings
            self.initDeltaMatrixes()
            #print((str(i)))

            for n in range(self.stochaticGroup):
                # run the network
                self.getInput(self.trainInput[i+n])
                for m in range(1, len(self.Nodes)):
                    self.sumAndMult(m)
                    self.addBias(m)
                    self.scale(m)
                # get correct answer and network answer
                networkGuess = np.argmax(self.nodeMatrix[len(self.Nodes)-1][0])
                self.setCorrect(networkGuess)
                #print(networkGuess)
                
                for L in range(self.D, -1, -1):
                    # set costs per node
                    self.setCost(L)
                    # increment delta bias
                    for j in range(self.Nodes[L+1]):
                        self.deltaBiasMatrix[L][0][j] = self.deltaBiasMatrix[L][0][j] + self.deltaBias(L,j)
                        for k in range(self.Nodes[L]):
                            self.deltaWeightMatrix[L][k][j] = self.deltaWeightMatrix[L][k][j] + self.deltaWeight(L,j,k)

            # add delta's to weights/biases
            self.weightMatrix = self.weightMatrix + self.deltaWeightMatrix
            self.biasMatrix = self.biasMatrix + self.deltaBiasMatrix
            print("Stochatic Group Complete")

        # run on both forward and backprop
        self.totalTests = 0
        self.totalRight = 0
        for i in range(self.numTests):
            self.getInput(self.testInput[i])
            for j in range (1,len(self.Nodes)):
                self.sumAndMult(j)
                self.addBias(j)
                self.scale(j)
            #check answer
            self.totalTests += 1
            networkGuess = np.argmax(self.nodeMatrix[len(self.Nodes)-1][0])
            if self.output[networkGuess] == self.correctOutput:
                self.totalRight += 1
                #print("Correct!")

        print("Total tests: " + str(self.totalTests))
        print("Total correct: " + str(self.totalRight))
        print("Final Error: " + str(1-(self.totalRight/self.totalTests)))

    #sets the cost per each node
    def setCost(self, Layer):
        for node in range(len(self.deltaCostMatrix[Layer])):
            if Layer == (self.D):
                self.deltaCostMatrix[Layer][0][node] = 2 * (self.nodeMatrix[Layer+1][0][node]-self.answerMatrix[node])
            else:
                for nextNode in range(len(self.deltaCostMatrix[Layer+1])):
                    self.deltaCostMatrix[Layer][0][node] = self.deltaCostMatrix[Layer][0][node] + (self.weightMatrix[Layer+1][node][nextNode] * self.dsigmoid(self.nodeMatrix[Layer+1][0][nextNode]) * self.deltaCostMatrix[Layer+1][0][nextNode])
    # gets the delta bias given a node
    def deltaBias(self, Layer, node):
        return (1)*(self.dsigmoid(self.nodeMatrix[Layer+1][0][node]))*(self.deltaCostMatrix[Layer][0][node])
    # gets the weight bias given a layer, node, and ndoe from
    def deltaWeight(self, Layer, node, fromNode):
        x = (self.nodeMatrix[Layer][0][fromNode])
        y = (self.dsigmoid(self.nodeMatrix[Layer+1][0][node]))
        z = (self.deltaCostMatrix[Layer][0][node])
        a = x * y * z
        return x

    # takes the input from the input set and prepares the network.
    # the second part of the tuple is the correct output
    def getInput(self, inputSet):

        #FOR MNIST ONLY
        pixels = []
        for i in range(28):
            for j in range(28):
                pixels.append(inputSet[0][i][j])
        ###########

        for i in range(self.N):
            self.nodeMatrix[0][0][i]=pixels[i]
        self.correctOutput = inputSet[1]
    # multiples and adds all the weights and previous nodes
    # stores them in the next node
    def sumAndMult(self, layer):
        #print("Summing input*weight")
        self.nodeMatrix[layer] = np.matmul(self.nodeMatrix[layer-1], self.weightMatrix[layer-1])
    # adds the bias to the weighted value
    def addBias(self, layer):
        #print("Applying Bias")
        self.nodeMatrix[layer] = self.nodeMatrix[layer] + self.biasMatrix[layer-1]
    # scales the function sigmoid or ReLU
    def scale(self, layer):
        #print("Scaling node")
        for i in range(self.Nodes[layer]):
            #print(str(layer) + ' ' + str(i))
            self.nodeMatrix[layer][0][i] = self.sigmoid(self.nodeMatrix[layer][0][i])
            #print(str(layer) + ' ' + str(i))

    # sigmoid, used to easily calculate sigmoid
    def sigmoid(self, x):
        return 1/(1+np.exp(x))
    # dsigmoid, returns the derivative of the sigmoid function for a given x
    # not a true delta sigmoid as we already have the sigmoid'ed value, using the dsigmoid, we can simplify this functio
    def dsigmoid(self, x):
        return x * (1-x)

