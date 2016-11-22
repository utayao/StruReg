# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
from Extractor import FcnExtractor
from proj_utils.local_utils import batchflow
from keras.utils import  generic_utils

from keras import backend as K
import matplotlib.pyplot as plt
import deepdish as dd

def get_wight_mask(Y_batch, params):
    # gt should be nsample*2*w*len
    label = Y_batch[:,0:-1,:,:]
    mean_label = np.mean(np.mean(label, axis = -1, keepdims=True), axis=-2, keepdims=True)
    mask = params['beta']*label + params['alpha']*mean_label
    Y_batch[:,-1::,:,:] = mask
    return Y_batch
def get_mean_std(StruExtractor,meanstd):
    if meanstd == 1:
       thismean, thisdev = StruExtractor.calMeanDevPts()
    if meanstd == 2:
        thismean, thisdev = StruExtractor.calMeanDevPts()
        thismean.fill(np.mean(thismean))
        thisdev.fill(np.mean(thisdev))
    if meanstd == 0:
        thismean = np.zeros((1,))
        thisdev = np.ones((1,))
    return thismean, thisdev

def get_model(buildmodel = None, arctecurepath=None, rebuildmodel=1,reuseweigths = 0, 
              img_channels=3, lr=1, loss='mse',weightspath =None):
    if not os.path.isfile(arctecurepath) or  rebuildmodel == 1:
       strumodel = buildmodel(img_channels,lr = lr, loss = loss)
       
       if reuseweigths == 1 and os.path.isfile(weightspath):
          strumodel.load_weights(weightspath)
    else:
        pass

def get_validation_double(trainparams, vpsize=135, lvpsize=135, img_channels=3, label_channels = 1, maxpatch=1):
    validation_params= trainparams.copy()
    validation_params['get_validation'] = True
    validation_params['maxpatch'] = maxpatch
    validation_params['patchsize'] = vpsize
    validation_params['labelpatchsize']= lvpsize
    validation_params['rotatepool']  = [0]

    validation_extractor = FcnExtractor(validation_params)
    validationMatinfo = validation_extractor.getMatinfo_volume()
    valid_datainfo = validationMatinfo['datainfo']
    Totalnum = valid_datainfo['Totalnum']
    valid_batch = np.zeros((Totalnum, valid_datainfo['inputdim']))
    valid_label = np.zeros((Totalnum, valid_datainfo['outputdim']))
    valid_secondlabel = np.zeros((Totalnum, valid_datainfo['outputdim']))
    validation_extractor.getOneDataBatch_stru(np.arange(Totalnum), thisbatch=valid_batch, 
                                              thislabel=valid_label, secondlabel = valid_secondlabel)

    valid_batch = np.reshape(valid_batch, (-1,vpsize, vpsize, img_channels ))
    valid_label = np.reshape(valid_label, (-1,lvpsize,lvpsize, label_channels ))
    valid_secondlabel = np.reshape(valid_secondlabel, (-1,lvpsize,lvpsize, label_channels ))

    valid_batch = np.transpose(valid_batch, (0, 3,1,2))
    valid_label = np.transpose(valid_label, (0, 3,1,2))
    valid_secondlabel = np.transpose(valid_secondlabel, (0, 3,1,2))


    return valid_batch, valid_label, valid_secondlabel

def train_blocks_doubleouputs(params):
    strumodel = params['strumodel']
    datainfo =  params['datainfo']
    labelpatchsize = params['labelpatchsize']
    patchsize = params['patchsize']
    label_channels, img_channels = params['label_channels'], params['img_channels']
    batch_size = params['batch_size']
    use_weighted = params['use_weighted']

    best_weightspath = params['best_weightspath']
    arctecurepath = params['arctecurepath']
    weightspath =   params['weightspath']
    #paramspath  = params['paramspath']
    weight_params = params['weight_params']
    savefre =  params['savefre']
    StruExtractor = params['StruExtractor']
    maxepoch = params['maxepoch']
    refershfreq = params['refershfreq']
    best_score = params['best_score']
    tolerance = params['tolerance']
    worseratio = params['worseratio']

    validation_batch_size = params['validation_batch_size']
    valid_batch, valid_label = params['valid_batch'], params['valid_label']
    chunknum = params['chunknum']
    nb_class = params['nb_class']
    thismean = params['thismean']
    thisdev  = params['thisdev']
    show_progress = params['show_progress']
    meanstd = 0
    count_ = 0

    thisbatch = np.zeros((chunknum,datainfo['inputdim']))
    thislabel = np.zeros((chunknum,datainfo['outputdim']))
    secondlabel = np.zeros((chunknum,datainfo['outputdim']))

    for epochNumber in range(maxepoch):
        if np.mod(epochNumber+1, refershfreq) == 0:
            Matinfo = StruExtractor.getMatinfo_volume() #call this function to generate nece info
            thismean, thisdev = get_mean_std(StruExtractor, meanstd)
            datainfo = Matinfo['datainfo']
        Totalnum = datainfo['Totalnum']
        totalIndx = np.random.permutation(np.arange(Totalnum))

        numberofchunk = (Totalnum + chunknum - 1) // chunknum   #the floor
        chunkstart = 0
        progbar = generic_utils.Progbar(Totalnum)

        progbar = generic_utils.Progbar(Totalnum)
        for chunkidx in range(numberofchunk):
            thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
            thisInd = totalIndx[chunkstart: chunkstart + thisnum]
            StruExtractor.getOneDataBatch_stru(thisInd, thisbatch[0:thisnum,:], thislabel[0:thisnum,:], secondlabel[0:thisnum,:])
            chunkstart += thisnum
            BatchData = thisbatch[0:thisnum,:].astype(K.FLOATX)
            BatchLabel = thislabel[0:thisnum,:].astype(K.FLOATX)
            BatchSecondLabel = secondlabel[0:thisnum,:].astype(K.FLOATX)

            if nb_class == 2 and  labelpatchsize == 1:
                BatchLabel = np.concatenate([BatchLabel, 1- BatchLabel], axis = -1)

            #---------------Train your model here using BatchData------------------
            BatchData -= thismean
            BatchData /= thisdev

            BatchData = np.reshape(BatchData, (-1,patchsize, patchsize, img_channels ))
            BatchLabel = np.reshape(BatchLabel, (-1,patchsize, patchsize, label_channels ))
            BatchSecondLabel = np.reshape(BatchSecondLabel, (-1,patchsize, patchsize, label_channels ))

            BatchData = np.transpose(BatchData, (0, 3,1,2))
            BatchLabel = np.transpose(BatchLabel, (0, 3,1,2))
            BatchSecondLabel = np.transpose(BatchSecondLabel, (0, 3,1,2))
            
            print('Training--Epoch--%d----chunkId--%d', (epochNumber, chunkidx))


            for X_batch, Y_batch, sec_batch in batchflow(batch_size, BatchData, BatchLabel,  BatchSecondLabel):
                if use_weighted == 0:
                    loss = strumodel.train_on_batch({'input': X_batch}, {'output_mask': Y_batch[:,0:-1,:,:],
                                                         'output_seg': sec_batch[:,0:-1,:,:]})
                else:
                    Y_batch = get_wight_mask(Y_batch, weight_params)
                    loss = strumodel.train_on_batch({'input': X_batch}, {'output_mask': Y_batch,
                                                      'output_seg': sec_batch})
                loss = np.mean(loss)
                assert not np.isnan(loss) ,"nan error"
            
            progbar.add(BatchData.shape[0], values = [("train loss", loss)])
            
            if np.mod(chunkidx, savefre) == 0:
                if use_weighted == 0:
                    valid_loss = strumodel.evaluate({'input': X_batch}, {'output_mask': Y_batch[:,0:-1,:,:],
                                                     'output_seg': sec_batch[:,0:-1,:,:]}, batch_size = validation_batch_size)
                else:
                    valid_label = get_wight_mask(valid_label, weight_params)
                    valid_loss = strumodel.evaluate({'input': X_batch}, {'output_mask': Y_batch,
                                                     'output_seg': sec_batch}, batch_size=validation_batch_size)
                valid_loss = np.mean(valid_loss)
                
                print('\nTesting loss: {}, best_score: {}'.format(valid_loss, best_score))
                if valid_loss <=  best_score:
                    best_score = valid_loss
                    print 'update to new best_score:', best_score
                    best_weight = strumodel.get_weights()
                    strumodel.save_weights(best_weightspath,overwrite = 1)
                    count_ = 0
                else:
                    count_ = count_ + 1
                    if valid_loss - best_score  > best_score * worseratio: 
                        strumodel.set_weights(best_weight)
                        print('weights have been reset to best_weights!')
                if count_ >= tolerance:
                    assert 0, 'performance not imporoved for so long'              
                json_string = strumodel.to_json()
                open(arctecurepath, 'w').write(json_string)
                strumodel.save_weights(weightspath,overwrite = 1)

            ndim = 1
            if show_progress == 1:
                testingbatch = BatchData[ndim:ndim+1,...]
                testinglabel = strumodel.predict({'input': testingbatch})[0][0]
                testingSeg   = strumodel.predict({'input': testingbatch})[1][0]
                True_seg = np.reshape(BatchSecondLabel[ndim,...],(label_channels, labelpatchsize, labelpatchsize))[0]
                testingTrue = np.reshape(BatchLabel[ndim,...],(label_channels, labelpatchsize, labelpatchsize))[0]

                plt.subplot(3,2,1)
                plt.imshow(np.reshape(testingTrue,(labelpatchsize, labelpatchsize)))
                plt.subplot(3,2,2)
                plt.imshow(np.reshape(testinglabel,(labelpatchsize, labelpatchsize)))
                plt.subplot(3,2,3)
                plt.imshow(testingbatch[0,1,...])
                plt.subplot(3,2,4)
                plt.imshow(np.reshape(testingSeg,(labelpatchsize, labelpatchsize)))
                plt.subplot(3,2,5)
                plt.imshow(np.reshape(True_seg,(labelpatchsize, labelpatchsize)))
                plt.show()
        strumodel.save_weights(weightspath,overwrite = 1)            
