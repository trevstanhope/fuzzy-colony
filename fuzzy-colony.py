"""
Fuzzy Colony
Application of Fuzzy maths for monitoring and controlling a "colony"
"""

__author__ = 'Trevor Stanhope'
__version__ = 0.1

import skfuzzy
import sklearn
import pymongo
from datetime import datetime, timedelta
import random
import numpy as np

class Colony:
    
    # Initialize
    def __init__(self, config):
        self.config = config
        self._db()
    
    # DB initializer
    def _db(self):
        addr = self.config['mongo_addr']
        port = self.config['mongo_port']
        dbname = self.config['mongo_dbname']
        client = pymongo.MongoClient(addr, port)
        self.db = client[dbname]
    
    # Add sample point to dataset
    def add_sample(self, sample):
        for p in self.config['parameters']:
            try:
                sample[p]
            except KeyError as e:
                print str(e)
        
        col = self.config['mongo_col']
        _id = self.db[col].insert(sample)
        assert _id is not None
                
    # Query Samples
    def query_samples(self, t_a, t_b):
        """
        t_a : datetime timestamp
        t_b : datetime timestamp
        """
        dataset = []
        name = self.config['mongo_dbname']
        parameters = self.config['parameters']
        matches = self.db[name].find({'time':{'$gt': t_a, '$lt' : t_b}})
        for m in matches:
            s = [m[p] for p in parameters]
            dataset.append(s)
        return np.array(dataset)
    
    # Cluster a subsection of the data
    def cluster_data(self, data, c=1, m=1.00, err=1.0, maxiter=1):
        """
        data : 2d array, size (S, N)
            Data to be clustered.  N is the number of data sets; S is the number
            of features within each sample vector.
        c : int
            Desired number of clusters or classes.
        m : float
            Array exponentiation applied to the membership function U_old at each
            iteration, where U_new = U_old ** m.
        error : float
            Stopping criterion; stop early if the norm of (U[p] - U[p-1]) < error.
        maxiter : int
        Maximum number of iterations allowed.
        """
        (cntr, U, U0, d, Jm, p, fpc) = skfuzzy.cmeans(data, c, m, err, maxiter) #!FIXME the source of this function is broken
        return cntr

if __name__ == '__main__':
    config = {
        'mongo_dbname' : 'test',
        'mongo_addr' : '127.0.0.1',
        'mongo_port' : 27017,
        'mongo_col' : 'test',
        'parameters' : {
            'temp_int' : {
                'units' : 'Celcius'
            },
            'temp_ext' : {
                'units' : 'Celcius'
            },
            'freq_int' : {
                'units' : 'Hertz'
            },
            'humidity_int' : {
                'units' : 'Relative Humidity'
            },
            'humidity_ext' : {
                'units' : 'Relative Humidity'
            }
        }
    }
    test = Colony(config)
    while True:
        try:
            # Test Sample
            pt = {
                'time' : datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
                'temp_int' : random.random(),
                'temp_ext' : random.random(),
                'freq_int' : random.random(),
                'humidity_int' : random.random(),
                'humidity_ext' : random.random()
                }
            test.add_sample(pt)
            
            # Test Query
            t_a = datetime.now() - timedelta(hours = 2)
            t_b = datetime.now()
            data = test.query_samples(t_a, t_b)
            
            # Test Cluster
            if data is not None:
                cntr = test.cluster_data(data)
                print cntr
        except KeyboardInterrupt:
            break
