# -*- coding: utf-8 -*-

from keras import backend as K

def elu(alpha=1):
    def f(x):
        pos = K.relu(x)
        neg = (x - K.abs(x)) * 0.5
        return pos + alpha * (K.exp(neg) - 1.)
    return f