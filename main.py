# Created 14 September 2020
# last updated 24 Sep 2020
###################################
# driver of the network
# gets data in and pushes it back out
######################################

from network import Network as nn

if __name__ == "__main__":
    myNetwork = nn()
    myNetwork.setNetworkSize()
    print(myNetwork)
    myNetwork.initMatrix()
    myNetwork.runNetwork()