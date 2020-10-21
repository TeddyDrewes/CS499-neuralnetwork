# created 24 Sep 2020, Teddy Drewes
# last update 21 Oct 2020
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
        self.iterations = 10000
        self.input=[]
        self.correctOutput = 0
        self.output= []
        self.backprop = True
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
        #self.iterations = int(input("Number of Training Iterations: "))
        #networkType = int(input("Back-propagation (0) or only Forward-propagation (1)? "))
        #if networkType == 0:
        #    self.backprop = True
        #else:
        #    self.backprop = False
        self.setInputMNIST(self.iterations)
        self.setOutput(self.M)

    #reads in the number of training sets to be used
    def setInputMNIST(self, numInputSets):
        # implement code here for this/can change based on data set
        print("setting NMIST input up")
        myMNIST = MNISTdd()
        myMNIST.runDriver()
        for i in range(numInputSets):
            self.input.append((myMNIST.data_dict['test images'][i],myMNIST.data_dict['test labels'][i]))


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

    # runs the network
    def runNetwork(self):
        for i in range(self.iterations):
            self.getInput(self.input[i]) #change this line to the input being used
            for j in range (1,len(self.Nodes)):
                self.sumAndMult(j)
                self.addBias(j)
                self.scale(j)
            #check answer
            self.totalTests += 1
            networkGuess = np.argmax(self.nodeMatrix[len(self.Nodes)-1][0])
            if self.output[networkGuess] == self.correctOutput:
                self.totalRight += 1
                print("Correct!")
            #backprop
            if self.backprop == True:
                #do backprop

        print("Total tests: " + str(self.totalTests))
        print("Total correct: " + str(self.totalRight))
        print("Final Error: " + str(1-(self.totalRight/self.totalTests)))

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
            self.nodeMatrix[layer][0][i] = (1/(1+np.exp(-(self.nodeMatrix[layer][0][i]))))
            #print(str(layer) + ' ' + str(i))
