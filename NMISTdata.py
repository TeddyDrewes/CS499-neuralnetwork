# created 21 Oct 2020, Teddy Drewes
# last update 21 Oct 2020
# MNIST data extracted using code from https://github.com/Ghosh4AI/Data-Processors/blob/master/MNIST/MNIST_Loader.ipynb
# Use is authorized
###############################################
# Extracts the code from MNIST and puts it in a useable format
###############################################

import os
import urllib.request
import gzip
import shutil
import codecs
import numpy

class MNISTdataDriver:

    def __init__(self):
        self.datapath = '../NMIST/'
        self.data_dict = {}

    def runDriver(self):
        self.downloadData()
        self.extractData()
        self.parseData()


    def downloadData(self):

        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

        for url in urls:
            filename = url.split('/')[-1]

            if os.path.exists(self.datapath+filename):
                print(filename, ' already exists')
            else:
                print('Downloading ', filename)
                urllib.request.urlretrieve(url,self.datapath+filename)

        print("All Files available")
    def extractData(self):
        files = os.listdir(self.datapath)
        for file in files:
            if file.endswith('gz'):
                print('Extracting ', file)
                with gzip.open(self.datapath+file, 'rb') as f_in:
                    with open(self.datapath+file.split('.')[0], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        print("All Files Extracted")
    def parseData(self):
        files = os.listdir(self.datapath)

        for file in files:
            if file.endswith('ubyte'):
                print('Reading ', file)
                with open(self.datapath+file, 'rb') as f:
                    data = f.read()
                    type = self.get_int(data[:4])
                    length = self.get_int(data[4:8])
                    if type == 2051:
                        category = "images"
                        num_rows = self.get_int(data[8:12])
                        num_cols = self.get_int(data[12:16])
                        parsed = numpy.frombuffer(data,dtype = numpy.uint8,offset = 16)
                        parsed = parsed.reshape(length,num_rows,num_cols)
                    elif (type == 2049):
                        category = 'labels'
                        parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8)
                        parsed = parsed.reshape(length)

                    if length == 10000:
                        set = 'test'
                    elif length == 60000:
                        set = 'train'

                    self.data_dict[set+' '+category] = parsed

        print("All data parsed and ready to be used")

    #converts 4 bytes to int
    def get_int(self, bytes):
        return int(codecs.encode(bytes, 'hex'), 16)



