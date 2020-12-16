import tensorflow as tf
import numpy as np
import random
import os

def set_seed_globally(seed_value=0):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
