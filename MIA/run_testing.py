from __future__ import absolute_import
import os
import sys
os.environ['THEANO_FLAGS'] = 'device=gpu2,optimizer=fast_run,force_device=True, allow_gc=True'


projroot = os.path.join('..')
homeroot = os.path.join('..','..','..')
testingimageroot = os.path.join(projroot,'Data', 'TestingData')

keras_folder = os.path.join(homeroot,'OwnCloud', 'Code', 'keras-1')
sys.path.insert(0, keras_folder)
sys.path.insert(0, os.path.join(keras_folder,'keras'))
sys.path.insert(0, os.path.join(keras_folder,'layers'))
sys.path.insert(0, projroot)
sys.path.insert(0, os.path.join(projroot, 'proj_utils') )

dataroot = os.path.join(projroot, 'Data')
modelroot = os.path.join(projroot, 'Data', 'Model')


import numpy as np
import deepdish as dd
from testingclass import runtestImg
from proj_utils.keras_utils import elu

pooling = 'avg'#'avg'
modelmarker = 'residule_fcn_'
weights_name = 'weights.h5'
nf=32
from kerasOneModel import buildMIAResidule as buildmodel
activ = elu(alpha=1.0) 
last_activ = 'relu'  

params_name  = 'params.h5'

sameModel = True
strumodel = None
if __name__ == "__main__":
    #Probrefresh = 0
    Seedrefresh = 1
    thresh_pool = np.arange(0.0,0.7,0.05)
    lenpool = [5, 6, 7, 8,9,10] #[3,5,7] #for cell detection
    
    steppool = [1] 
    showmap = 1
    showseed = 1
    batchsize = 1280
    resizeratio = 1
    printImg = 1

    testingParam = {}
    testingParam['windowsize'] = None
    testingParam['batch_size'] = 2
    testingParam['fixed_window'] = False
    testingParam['board'] = 30
    testingParam['step_size'] = None
    
                    # testingset, ImgExt, trainingset, modelsubfolder, testtype , test, evaluate ,Probrefresh
    testingpool = [ 
                    ('testPhasecontrast_convert' , ['.tif'], 'trainPhasecontrast_convert',  modelmarker + pooling, 'fcn',True, False, True),
                  ]
    for tetstingset in testingpool: 
        testtype = 'fcn'
        testingset, ImgExt, trainingset, modelsubfolder, testtype , test, evaluate ,Probrefresh = tetstingset

        weights_name_noext, _ = os.path.splitext(weights_name)
        metric_name  = modelsubfolder + 'metric' + '_' + weights_name_noext
        resultmask   = modelsubfolder  + '_' + weights_name_noext

        testingimagefolder = os.path.join(testingimageroot, testingset)

        savefolder = os.path.join(testingimagefolder, resultmask)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        if sameModel != True or strumodel is None:
            modelfolder = os.path.join(modelroot,trainingset, modelsubfolder)
            meandevpath = os.path.join(modelfolder, 'meanstd.h5')
            arcpath = os.path.join(modelfolder, 'arc.json')
            weightspath = os.path.join(modelfolder,weights_name)
            paramspath = os.path.join(modelfolder, 'params.h5')
            ModelDict = {}
            meandev = dd.io.load(meandevpath)
            trainparam = dd.io.load(paramspath)
            #trainparam = None
            if test:
                strumodel = buildmodel(img_channels=3, activ=activ, last_activ='relu',pooling=pooling, nf=nf,make_predict = True)
                strumodel.load_weights(weightspath)
            else:
                strumodel = {}

        ModelDict['params'] = trainparam
        ModelDict['thismean'] = meandev['thismean']
        ModelDict['thisdev'] = meandev['thisdev']
        ModelDict['model'] = strumodel

        classparams = {}
        classparams['ImgDir'] = testingimagefolder
        classparams['savefolder'] = savefolder
        classparams['resultmask'] = resultmask
        classparams['ImgExt'] =  ImgExt
        classparams['patchsize']   = ModelDict['params']['patchsize']
        classparams['labelpatchsize']   = ModelDict['params']['labelpatchsize']

        classparams['resizeratio'] = resizeratio

        classparams['model'] =     ModelDict['model']
        #classparams['steppool'] =  [ModelDict['params']['labelpatchsize']]
        classparams['steppool'] = steppool
        classparams['channel'] =   ModelDict['params']['channel']
        classparams['thismean']  = ModelDict['thismean']
        classparams['thisdev']  =  ModelDict['thisdev']

        classparams['Probrefresh']  =  Probrefresh
        classparams['Seedrefresh']  =  Seedrefresh

        classparams['lenpool']  =  lenpool
        classparams['showmap']  =  showmap
        classparams['showseed'] = showseed
        classparams['batchsize']  =  batchsize
        classparams['test_type'] = testtype

        tester = runtestImg(classparams)
        if test:
            print('start testing!')
        tester.runtesting(**testingParam)

        
