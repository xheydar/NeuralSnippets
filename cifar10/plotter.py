import numpy as np 
import pickle
from matplotlib import pyplot as pp 
pp.ion()

class plotter :
    def __init__( self, tag ):
        
        data_path = 'results_%s.pkl' % ( tag )

        with open( data_path, 'rb') as ff :
            data = pickle.load(ff)

        test_acc = data['test_acc']
        epoch_indices = np.arange( len(test_acc) ) + 1

        pp.figure()
        pp.plot( epoch_indices, test_acc )
        pp.title( tag )
        pp.ylabel('Test Accuracy')
        pp.xlabel('Epoch Index')
        pp.grid()
