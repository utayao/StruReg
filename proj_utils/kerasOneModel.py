from __future__ import absolute_import

from keras import backend as K
from keras.layers import Input,Dropout, merge,LocalResponseNorm,BatchNormalization, Dense,Lambda
from keras.layers import TiedHighway, Flatten, Reshape, Activation

from keras.layers.convolutional import  MaxPooling2D, AveragePooling2D,UpSampling2D, Resize2D,Convolution2D # BLUpSampling2D,
from keras.optimizers import Adadelta  ,SGD, RMSprop, adam
from keras.layers.advanced_activations import ELU

from keras.regularizers import l2
from keras.models import Model

def _scaling_output_shape(input_shape, cls):
    return input_shape
def _scaling(x, scaling):
    return scaling*x
scaling_layer = Lambda(_scaling, output_shape =_scaling_output_shape, arguments= {'scaling':0.5} )    


def buildFCRNAModel(img_channels=3, lr = 0.01,weight_decay = 0.0005, loss='mse',activ='relu', last_activ='sigmoid', **kwargs):
    main_input = Input(shape=(img_channels, None, None), name='input')
    conv_1 = Convolution2D(32,3,3, border_mode = 'same', activation= activ, init='orthogonal',name='conv_1',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(main_input)
    max_0 = MaxPooling2D(pool_size = (2,2))(conv_1)

    conv_2 = Convolution2D(64,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_2',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(max_0)
    max_1 = MaxPooling2D(pool_size = (2,2))(conv_2)

    conv_3 = Convolution2D(128,3,3, border_mode = 'same', activation= activ, init='orthogonal',name='conv_3',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(max_1)  # 25
    max_2 = MaxPooling2D(pool_size = (2,2))(conv_3)    

    conv_4 = Convolution2D(512,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_4',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(max_2)  #
                                                          # 12
    upsamp_1 = UpSampling2D((2,2))(conv_4)
    resize_1 = Resize2D(input_tensor = conv_3)(upsamp_1)
    relu_res = Activation('relu')(resize_1)
    deconv_1 = Convolution2D(128,3,3, border_mode = 'same', activation='linear', init='orthogonal',name='deconv_1',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(relu_res)

    upsamp_2 = UpSampling2D((2,2))(deconv_1)
    resize_2 = Resize2D(input_tensor = conv_2)(upsamp_2)
    relu_res = Activation('relu')(resize_2)
    deconv_2 = Convolution2D(64,3,3, border_mode = 'same', activation='linear',init='orthogonal',name='deconv_2',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(relu_res)

    upsamp_3 = UpSampling2D((2,2))(deconv_2)
    resize_3 = Resize2D(input_tensor=main_input)(upsamp_3)
    relu_res = Activation('relu')(resize_3)
    deconv_3 = Convolution2D(32,3,3, border_mode = 'same', activation='linear',init='orthogonal',name='deconv_3',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(relu_res)

    last_conv = Convolution2D(1,3,3, border_mode = 'same', activation=last_activ,init='orthogonal', name= 'output_mask',
                                       W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(deconv_3)

    model = Model(input=[main_input], output=[last_conv])
    #opt = SGD(lr=lr, decay= 1e-6, momentum=0.9,nesterov=False)
    opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-06,clipvalue=0.5)
    #opt = adam(lr=lr)
    model.compile(loss={'output_mask': loss }, optimizer=opt)
    return model


def buildFCRNBModel(img_channels=3, lr = 0.01,weight_decay = 0.0005, loss='mse',activ='relu', 
                    last_activ='sigmoid',**kwargs):
    main_input = Input(shape=(img_channels, None, None), name='input')
    conv_1 = Convolution2D(32,3,3, border_mode = 'same', activation= activ, init='orthogonal',name='conv_1',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(main_input)

    conv_2 = Convolution2D(64,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_2',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(conv_1)
    max_1 = MaxPooling2D(pool_size = (2,2))(conv_2)

    conv_3 = Convolution2D(128,3,3, border_mode = 'same', activation= activ, init='orthogonal',name='conv_3',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(max_1)  # 25
    mp = MaxPooling2D(pool_size = (2,2))(conv_3)                             
    conv_4 = Convolution2D(256,5,5, border_mode = 'same', activation=activ, init='orthogonal',name='conv_4',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(mp)  #
    conv_5 = Convolution2D(256,3,3, border_mode = 'same', activation=activ, init='orthogonal',name='conv_5',
                                 W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(conv_4)  #

#    upsamp_1 = UpSampling2D((2,2))(conv_5)
#    resize_1 = Resize2D(input_tensor=conv_3)(upsamp_1)
#    relu_res = Activation('relu')(resize_1)
#    deconv_1 = Convolution2D(256,5,5, border_mode = 'same', activation='linear', init='orthogonal',name='deconv_1',
#                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(relu_res)
#
#    upsamp_2 = UpSampling2D((2,2))(deconv_1)
#    resize_2 = Resize2D(input_tensor=conv_2)(upsamp_2)
#    relu_res = Activation('relu')(resize_2)
#    deconv_2 = Convolution2D(1,3,3, border_mode = 'same', activation='linear',init='orthogonal',name='deconv_2',
#                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(relu_res)

    upsamp_1 = UpSampling2D((2,2))(conv_5)
    resize_1 = Resize2D(input_tensor=conv_3)(upsamp_1)
    #relu_res = Activation('relu')(resize_1)
    deconv_1 = Convolution2D(256,5,5, border_mode = 'same', activation='relu', init='orthogonal',name='deconv_1',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(resize_1)

    upsamp_2 = UpSampling2D((2,2))(deconv_1)
    resize_2 = Resize2D(input_tensor=conv_2)(upsamp_2)
    #relu_res = Activation('relu')(resize_2)
    deconv_2 = Convolution2D(1,3,3, border_mode = 'same', activation='relu',init='orthogonal',name='deconv_2',
                                   W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(resize_2)
                                   
                                   
    last_conv = Convolution2D(1,3,3, border_mode = 'same', activation=last_activ,init='orthogonal', name= 'output_mask',
                                       W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(deconv_2)

    model = Model(input=[main_input], output=[last_conv])
    #opt = SGD(lr=lr, decay= 1e-6, momentum=0.9,nesterov=False)
    opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-06,clipvalue=0.5)
    #opt = adam(lr=lr)
    model.compile(loss={'output_mask': loss }, optimizer=opt)
    return model


def up_tensor(input, reference_tensor=None,merge_tensor=True, base_name = 'up_tensor'):
    upsamp_0 = UpSampling2D((2,2))(input)
    resize_0 = Resize2D(input_tensor=reference_tensor)(upsamp_0)
    if merge_tensor == True:
        input_tensor  = merge([resize_0, reference_tensor], mode='concat', concat_axis=1)
    else:
        input_tensor = resize_0
    return input_tensor

def bn_conv(input, filters = 64, kernel=(3,3),subsample=(1,1), base_name='bn_conv', activ='relu',conv_activ = 'linear', 
            bn = True, weight_decay = 1e-7, init='orthogonal',trainable = True):
    if bn == True:    
        norm = BatchNormalization(mode=0, axis=1, name= base_name + '_bn1')(input)
    else:
        norm = input
    activ_value = Activation(activ, name= base_name + '_activ_1')(norm)
    conv_1 = Convolution2D(filters,kernel[0],kernel[1], border_mode = 'same', subsample=subsample,activation= conv_activ, init= init, name= base_name,
                           trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
    return conv_1

def up_conv(input, reference_tensor=None, filters = 64, kernel=(3,3), merge_tensor=True, base_name='up_bn_conv', 
            activ='relu', conv_activ = 'linear' ,weight_decay = 1e-7, trainable = True): 
    input_tensor = up_tensor(input, reference_tensor=reference_tensor,merge_tensor=merge_tensor, base_name = base_name)
    conv_value = bn_conv(input_tensor, filters = filters, kernel=kernel, base_name= base_name, activ=activ,
                         conv_activ = conv_activ, weight_decay = weight_decay)
    return conv_value

def Inception_residual(input, filters = 64, kernel=(3,3), slim_kernel= 5, base_name='block', activ='relu',mode="sum", 
                       scaling=0.5, weight_decay = 1e-7, init = 'orthogonal', trainable = True):
    droprate = 0.33
    norm = BatchNormalization(mode=0, axis=1, name= base_name + '_bn1')(input)
    activ_value = Activation(activ, name= base_name + '_activ_1')(norm)

    conv_1 = Convolution2D(filters,kernel[0],kernel[1], border_mode = 'same', activation= 'linear', init= init, name= base_name + '_conv_1',
                           trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
    dp = Dropout(droprate)(conv_1)
    #norm = BatchNormalization(mode=0, axis=1, name= base_name + '_bn2')(dp)
    activ_value = Activation(activ,name= base_name + '_activ_2')(dp)
    conv_1_1 = Convolution2D(filters,kernel[0],kernel[1], border_mode = 'same', activation='linear', init= init,name= base_name + '_conv_2',
                             trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
    transform = conv_1_1
    if slim_kernel is not None:
        slim_1 =  Convolution2D(filters,1, slim_kernel, border_mode = 'same', activation= 'linear', init= init, name= base_name + '_slim_1',
                           trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
        dp = Dropout(droprate)(slim_1)
        #norm = BatchNormalization(mode=0, axis=1, name= base_name + '_bn2')(dp)
        activ_value = Activation(activ,name= base_name + '_activ_3')(dp)
        slim_2 =  Convolution2D(filters, slim_kernel, 1,  border_mode = 'same', activation= 'linear', init= init, name= base_name + '_slim_2',
                           trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
        interConcate = merge([conv_1_1, slim_2], mode='concat', concat_axis=1)
        
        slim_out =  Convolution2D(filters, 1, 1,  border_mode = 'same', activation= 'linear', init= init, name= base_name + '_slim_2',
                           trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)

        transform = scaling_layer(slim_out, scaling)
    if mode == 'sum':
        merged = merge([input, transform], mode="sum")
    elif mode == 'concat':
        merged = merge([input, transform], mode='concat', concat_axis=1)
    elif mode == 'none':
        return transform
    else:
        assert 0, 'unknown merge mode'
    return merged

def fcn_residual(input, filters = 64, kernel=(3,3), base_name='block', activ='relu',mode="sum", weight_decay = 1e-7, 
                 bn=True, scaling = None, init = 'orthogonal', trainable = True):
    droprate = 0.33
    if bn == True:
        norm = BatchNormalization(mode=0, axis=1, name= base_name + '_bn1')(input)
    else:
        norm = input
    
    activ_value = Activation(activ, name= base_name + '_activ_1')(norm)
    conv_1 = Convolution2D(filters,kernel[0],kernel[1], border_mode = 'same', activation= 'linear', init= init, name= base_name + '_conv_1',
                           trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
    dp = Dropout(droprate)(conv_1)
    if bn == True:    
        norm = BatchNormalization(mode=0, axis=1, name= base_name + '_bn2')(dp)
    else:
        norm = dp
    activ_value = Activation(activ,name= base_name + '_activ_2')(norm)
    conv_1_1 = Convolution2D(filters,kernel[0],kernel[1], border_mode = 'same', activation='linear', init= init,name= base_name + '_conv_2',
                             trainable =trainable, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(activ_value)
    if scaling is not None:
        conv_1_1 = scaling_layer(conv_1_1, scaling)
    if mode == 'sum':
        merged = merge([input, conv_1_1], mode="sum")
    elif mode == 'concat':
        merged = merge([input, conv_1_1], mode='concat', concat_axis=1)
    elif mode == 'none':
        return conv_1_1
    else:
        assert 0, 'unknown merge mode'
    return merged

def buildMIAResidule(img_channels=3, lr = 0.01, weight_decay = 1e-7, loss='mse',activ='relu', bn = False, scaling=0.3,
                     nf = 32,  pooling = 'max',  last_activ='sigmoid',opt=None, make_predict=False):
    if opt is None: 
       opt = adam(lr=lr)
    droprate = 0.33
    
    if pooling == 'max':
        Pooling = MaxPooling2D #AveragePooling2D # ,
    elif pooling == 'avg':
        Pooling = AveragePooling2D
    else:
        assert 0, 'unkonwn pooling setting.'
    main_input = Input(shape=(img_channels, None, None), name='input')
    double_0 = bn_conv(main_input, filters=nf, kernel=(1,1), base_name='bn_double_0', activ=activ,  bn = bn, weight_decay = weight_decay) 
    block1 = fcn_residual(double_0, filters=nf, base_name='block_1', activ= activ ,bn = bn, scaling=scaling, weight_decay = weight_decay)  
    double_1 = bn_conv(block1, filters=nf*2, kernel=(1,1), base_name='bn_double_1', activ=activ,  bn = bn, weight_decay = weight_decay)                        
    maxpool = Pooling(pool_size = (2,2))(double_1)
    
    block2 = fcn_residual(maxpool, filters=nf*2, base_name='block_2', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)
    double_2 = bn_conv(block2, filters=nf*4, kernel=(1,1), base_name='bn_double_2', activ=activ,  bn = bn, weight_decay = weight_decay)                        
    maxpool = Pooling(pool_size = (2,2))(double_2)

    block3 = fcn_residual(maxpool, filters=nf*4, base_name='block_3', activ= activ,bn = bn, scaling=scaling, weight_decay = weight_decay)
    double_3 = bn_conv(block3, filters=nf*8, kernel=(1,1), base_name='bn_double_3', activ=activ,  bn = bn, weight_decay = weight_decay)                        
    maxpool = Pooling(pool_size = (2,2))(double_3)

    block4 = fcn_residual(maxpool, filters=nf*8, base_name='block_4', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)
    #double_4 = bn_conv(block4, filters=512, base_name='bn_double_4', activ=activ,weight_decay = weight_decay)                        
    maxpool = Pooling(pool_size = (2,2))(block4)

    block5 = fcn_residual(maxpool, filters=nf*8, base_name='block_5', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)                    

    up_2 = up_conv(block5, reference_tensor=block4, filters = nf*8,  kernel=(1,1), base_name='up_bn_conv_2')
    deconv_2 = fcn_residual(up_2, filters=nf*8, base_name='deconv_2', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)

    up_3 = up_conv(deconv_2, reference_tensor=block3, filters = nf*4,  kernel=(1,1), base_name='up_bn_conv_3')
    deconv_3 = fcn_residual(up_3, filters=nf*4, base_name='deconv_3', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)

    up_4 = up_conv(deconv_3, reference_tensor=block2, filters = nf*2,  kernel=(1,1), base_name='up_bn_conv_4')
    deconv_4 = fcn_residual(up_4, filters = nf*2, base_name='deconv_4', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)

    up_5 = up_conv(deconv_4, reference_tensor=block1, filters = nf,  kernel=(1,1), merge_tensor=False, base_name='up_bn_conv_5')
    deconv_5 = fcn_residual(up_5, filters=nf, base_name='deconv_5', activ= activ, bn = bn, scaling=scaling, weight_decay = weight_decay)

    last_conv = bn_conv(deconv_5, filters=1, kernel=(1,1), base_name='output_mask', activ=activ,  bn = bn,
                        conv_activ=last_activ, weight_decay = weight_decay) 

    model = Model(input=[main_input], output=[last_conv])
    #opt = SGD(lr=lr, decay= 1e-7, momentum=0.9,nesterov=True)
    #opt = Adadelta(lr=lr, rho=0.95, epsilon=1e-06,clipvalue=10)   
    model.compile(loss={'output_mask': loss}, optimizer=opt, make_predict= make_predict)
    return model

