# -*- coding: utf-8 -*-
from keras import backend as K
import numpy as np

def weighted_loss( base='mse'):
    '''
    The y_pred_mask has the mask in the last dimension, but the y_true does not have that
    y_true is (batch, channel, row, col), we need to permite the dimensio first
    '''   
    def loss(y_true_mask, y_pred):
        assert K.ndim(y_pred) <= 5, 'dimension larger than 5 is not implemented yet!!'
        if K.ndim(y_pred) == 3:
            y_true = K.permute_dimensions(y_true_mask[:,0:-1, :],(0,2,1))
            y_pred = K.permute_dimensions(y_pred,(0,2,1))
            y_mask = y_true_mask[:,-1,:]
        elif K.ndim(y_pred) == 4:
            y_true = K.permute_dimensions(y_true_mask[:,0:-1,:,:],(0,2,3,1))
            y_pred = K.permute_dimensions(y_pred, (0,2,3,1))
            y_mask = y_true_mask[:,-1,:,:]
        elif K.ndim(y_pred) == 5:
            y_true = K.permute_dimensions(y_true_mask[:,0:-1,:,:,:], (0,2,3,4,1))
            y_pred = K.permute_dimensions(y_pred, (0,2,3,4,1))
            y_mask = y_true_mask[:,-1,:,:,:]
        
        naive_loss = get(base)(y_true,y_pred)
        return naive_loss * y_mask
    return loss


def fcn_loss( base='mse'):
    '''
    The y_pred_mask has the mask in the last dimension, but the y_true does not have that
    y_true is (batch, channel, row, col),we need to permite the dimensio first
    '''   
    def loss(y_true, y_pred):
        assert K.ndim(y_pred) <= 5, 'dimension larger than 5 is not implemented yet!!'
        if K.ndim(y_pred) == 3:
            y_true = K.permute_dimensions(y_true,(0,2,1))
            y_pred = K.permute_dimensions(y_pred,(0,2,1))
            
        elif K.ndim(y_pred) == 4:
            y_true = K.permute_dimensions(y_true,(0,2,3,1))
            y_pred = K.permute_dimensions(y_pred, (0,2,3,1))
          
        elif K.ndim(y_pred) == 5:
            y_true = K.permute_dimensions(y_true, (0,2,3,4,1))
            y_pred = K.permute_dimensions(y_pred, (0,2,3,4,1))
        
        naive_loss = get(base)(y_true,y_pred)
        return naive_loss
    return loss
    
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.categorical_crossentropy(y_pred, y_true)


def sparse_categorical_crossentropy(y_true, y_pred):
    '''expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    '''
    return K.sparse_categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity

from generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')



# def squared_error(y_true, y_pred):
#     return K.square(y_pred - y_true)

# def binary_crossentropy(y_true, y_pred):
#     return K.binary_crossentropy(y_pred, y_true)


# def absolute_error(y_true, y_pred):
#     return K.abs(y_pred - y_true)


# def absolute_percentage_error(y_true, y_pred):
#     diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
#     return 100. * diff


# def squared_logarithmic_error(y_true, y_pred):
#     first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
#     second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
#     return K.square(first_log - second_log)


# def squared_hinge(y_true, y_pred):
#     return K.square(K.maximum(1. - y_true * y_pred, 0.))


# def hinge(y_true, y_pred):
#     return K.maximum(1. - y_true * y_pred, 0.)


# def categorical_crossentropy(y_true, y_pred):
#     '''Expects a binary class matrix instead of a vector of scalar classes.
#     '''
#     return K.categorical_crossentropy(y_pred, y_true)


# def poisson(y_true, y_pred):
#     return y_pred - y_true * K.log(y_pred + K.epsilon())

# # aliases
# se = SE = squared_error
# ae = AE = absolute_error
# ape = APE = absolute_percentage_error
# sle = SLE = squared_logarithmic_error
# bc= binary_crossentropy
# cc=categorical_crossentropy

# from generic_utils import get_from_module
# def get(identifier):
#     return get_from_module(identifier, globals(), 'objective')
