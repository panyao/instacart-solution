# -*- coding: utf-8 -*-

import sys
from pandas import HDFStore

def save_to_hdfs(variable, filename):
    store = HDFStore(filename + '.h5')
    store[filename] = variable  # save it
    store.close()
    
def load_from_hdfs(filename):
    store = HDFStore(filename + '.h5')
    variable = store[filename]  # save it
    store.close()
    return variable

# Get memory usage statistics
def show_memory():
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)