"""
Fuzzy Colony

Application of Fuzzy maths for honey bee colony health estimation
"""

__author__ = 'Trevor Stanhope'
__version__ = 0.1

import skfuzzy
import sklearn

class Colony:

    def __init__(self, config):
        self.config = config
        
    def estimate_health(self):
        return 

if __name__ == '__main__':
    config = {
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
