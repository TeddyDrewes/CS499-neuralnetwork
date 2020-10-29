# Created 14 September 2020
# last updated 21 Oct 2020
###################################
# driver of the network
# gets data in and pushes it back out
######################################

from network import Network as nn


def singleRun():
    myNetwork = nn()
    myNetwork.setNetworkSize()
    print(myNetwork)
    myNetwork.initMatrix()
    myNetwork.runNetwork()


def multRun():
    totalCorrect = 0
    totalRan = 0
    numIterations = int(input("How many iterations? "))
    myNetwork = nn()
    myNetwork.setNetworkSize()

    for i in range(numIterations):
        print("Iteration " + str(i))
        myNetwork.initMatrix()
        myNetwork.runNetwork()
        totalCorrect += myNetwork.totalRight
        totalRan += myNetwork.totalTests

    print("Total Simulations: " + str(numIterations))
    print("Total Tests Ran: " +  str(totalRan))
    print("Total Correct: " + str(totalCorrect))
    print("Average Correct: " + str(totalCorrect/numIterations))
    print("Average Error: " + str(1-(totalCorrect/totalRan)))


if __name__ == "__main__":
    runType = int(input("Single (1) or Multiple (2)? "))
    if runType == 1:
        singleRun()
    else:
        multRun()

