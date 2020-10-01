# created 24 Sep 2020, Teddy Drewes
# last update 24 Sep 2020
###############################################
# The network object
# Runs the network
###############################################

import numpy as np
import random

class Network:

    def __init__(self):
        self.N = 0
        self.M = 0
        self.D = 0
        self.Nodes = []
        self.biasRange = [0,0]
        self.weightRange = [0,0]

    def __str__(self):
        return  ('\nInputs: ' + str(self.N) + '\n' \
        'Outputs: ' + str(self.M) + '\n' \
        'Hidden Layers: ' + str(self.D) + '\n' \
        'Width of Layers Input-->Output ' + str(self.Nodes) + '\n' \
        'Bias Range: ' + str(self.biasRange[0]) + ' to ' + str(self.biasRange[1]) + '\n' \
        'Weight Range: ' + str(self.weightRange[0]) + ' to ' + str(self.weightRange[1]))

    def setNetworkSize(self):
        self.N = int(input("How many inputs are there? "))
        self.Nodes.append(self.N)
        self.M = int(input("How many possible outputs are there? "))
        self.D = int(input("How many hidden layers would you like? "))
        for i in range (self.D):
            self.Nodes.append(int(input("How many nodes in Layer " + str(i) + "? ")))
        self.Nodes.append(self.M)
        self.biasRange[0] = float(input("Minimum Bias value? "))
        self.biasRange[1] = float(input("Maximum Bias value? "))
        self.weightRange[0] = float(input("Minimum Weight value? "))
        self.weightRange[1] = float(input("Maximum weight value? "))

    def initMatrix(self):
        #define weight matrixes
        self.weightMatrix = []
        for i in range (1,len(self.Nodes)):
            self.weightMatrix.append(np.empty([self.Nodes[i],self.Nodes[i-1]],dtype=float))
        for mat in self.weightMatrix:
            for i in range (0,len(mat)):
                for j in range(0,len(mat[i])):
                    mat[i][j] = random.uniform(self.weightRange[0],self.weightRange[1])
        #print(self.weightMatrix)

        #define bias matrixes
        self.biasMatrix = []
        for i in range(1, len(self.Nodes)):
            self.biasMatrix.append(np.empty([self.Nodes[i], self.Nodes[i-1]], dtype=float))
        for mat in self.biasMatrix:
            for i in range(0, len(mat)):
                for j in range(0, len(mat[i])):
                    mat[i][j] = random.uniform(self.biasRange[0], self.biasRange[1])
        #print(self.biasMatrix)

        #define Node Matrixes
        self.nodeMatrix = []
        for i in range(len(self.Nodes)):
            self.nodeMatrix.append(np.empty([self.Nodes[i],1],dtype = float))
        print(self.nodeMatrix)
