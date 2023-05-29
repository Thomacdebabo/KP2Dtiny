"""Logging utilities for debugging
"""
import tensorflow as tf
from termcolor import colored

from functools import wraps
from time import time

def timing(f):
    """
    Timing decorator, prints elapsed time of a given function.
    :param f: function to time
    :return:
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print ('func:%r  took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

def checkNan(f):
    """
    Decorator to check tensor inputs for NaN values.
    :param f: function to check
    :return:
    """
    @wraps(f)
    def wrap(*args, **kw):
        for arg in args:
            if tf.is_tensor(arg):
                if test_nan(arg):
                    print("Encountered Nan value in", f.__name__)
        result = f(*args, **kw)
        return result
    return wrap
def test_nan(x):
    return tf.reduce_any(tf.math.is_nan(tf.cast(x, tf.float32)))


def printcolor_single(message, color="white"):
    """Print a message in a certain color"""
    print(colored(message, color))


def printcolor(message, color="white"):
    "Print a message in a certain color (only rank 0)"

    print(colored(message, color))
