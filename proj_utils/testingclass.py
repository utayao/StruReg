import os
import time
from scipy import ndimage
import scipy.io as sio
import numpy as np
import deepdish as dd
from  skimage.feature import peak_local_max
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_float
import shutil
from local_utils import *

class runtestImg(object):
    def __init__(self, kwargs):
        self.ImgDir = ''
        self.savefolder = ''
        self.ImgExt = ['.jpg']
        self.resultmask = ''
        self.patchsize = 48
        self.labelpatchsize = 48
        self.channel = 3
        self.model = None
        self.stridepool = [self.labelpatchsize]
        self.arctecurepath = ''
        self.thismean = ''
        self.thisdev = ''
        self.Probrefresh = 1
        self.Seedrefresh = 1
        self.thresh_pool = np.arange(0.05, 1, 0.05)
        self.lenpool = [8]
        self.show = 0
        self.batchsize = 1280
        self.batch_size = 256
        self.resizeratio = 1
        self.metric_name = 'metric'
        self.test_type = 'fcn'
        self.eva_path = 'eva_info.h5'
        self.ol_folder = 'ol_folder'
        self.testingInd = None
        self.steppool = [1]
        for key in kwargs:
            setattr(self, key, kwargs[key])
        #assert self.model != None,"Model does not exist!"
    def get_output(self, BatchData, batch_size = 256,**kwargs):
        '''
        This function is a general interface for getting output for a given testing batch
        BatchData is of size: batch_size * dimensional
        
        '''
        if self.test_type == 'cnn':
            BatchData = np.reshape(BatchData, (BatchData.shape[0],self.patchsize, self.patchsize, -1))
            BatchData = np.transpose(BatchData, (0, 3,1,2))
            thisprediction  =  self.model.predict({'input': BatchData},batch_size = batch_size)['output_mask']
            
        if self.test_type == 'fcn' or self.test_type == 'dilated':
            # for fcn, the testing batch data should be of size row*col*channel
            inputShape = BatchData.shape

            param = myobj()
            param.windowsize = 1000
            param.board = 40
            param.fixed_window = False
            param.step_size = None

            for key in kwargs:
                setattr(param, key, kwargs[key])

            assert len(BatchData.shape) == 3, 'input image must be 3 dimensional at this time'

            BatchData = np.transpose(BatchData, (2,0,1))
            thisprediction = self.get_fcn_split(BatchData, windowsize = param.windowsize, board = param.board, batch_size = batch_size,
                                                fixed_window= param.fixed_window, step_size= param.step_size)

        if self.test_type == 'fcn_nosplit':
            #in this case, BatchData is just image after pre_process_img, the preiction is simply a mask tensor
            # img should be [row, col, channel]
            inputShape = BatchData.shape

            if len(BatchData.shape) == 3:
                BatchData = BatchData[np.newaxis]
            #the input image need to be permuted to make sure [batch, channel, row, col]
            BatchData = np.transpose(BatchData, (0,3,1,2))
            thisprediction  =  self.model.predict({'input': BatchData},batch_size = batch_size)['output_mask']
            if thisprediction.shape[0] == 1: # because we want the output has the same shape wth input
                if len(inputShape) == 3:
                    thisprediction = thisprediction[0]
        return  thisprediction    
    
    def fusionfunc_stru(self, cnnprediction,VotingMap, predictindx, Rowsize, Colsize,filterRow, filterCol):
        [rowindx, colindx] = np.unravel_index(predictindx, (Rowsize, Colsize))
        testingstartAndEnd = CentralToOrigin(rowindx,colindx,filterRow,filterCol)
        testingRowStart = testingstartAndEnd['RS']
        testingColomnStart = testingstartAndEnd['CS']
        for predIndx in range(cnnprediction.shape[0]):
            #print predIndx
            thisPred = np.reshape(cnnprediction[predIndx, :], (filterRow, filterCol))
            VotingMap[testingRowStart[predIndx]:testingRowStart[predIndx]+filterRow,
                      testingColomnStart[predIndx]:testingColomnStart[predIndx]+filterCol] += thisPred
            
    def get_fcn_split(self, img, windowsize = 500, board = 30, batch_size = 4,fixed_window= False,step_size=None, **kwargs):
        '''
        This function is used to do prediction for fcn and it's variation'
        img should be of size (channel, row, col)
        '''
        assert  len(img.shape) == 3, "Input image must be have three dimension"

        AllDict, PackList =  split_img(img, windowsize=windowsize, board = board, fixed_window= fixed_window,step_size=step_size)
        output = np.zeros(img.shape[1:])

        for key in AllDict.keys():
            iden_list = AllDict[key]
            this_size = PackList[iden_list[0]][3]

            BatchData = np.zeros( (len(iden_list), img.shape[0]) + tuple(this_size) )
            for idx, iden in enumerate(iden_list):
                BatchData[idx,...] = PackList[iden][0] #store it into numpy array
            if self.test_type == "fcn-legency":
                thisprediction  =  self.model.predict({'input': BatchData},batch_size = batch_size)['output_mask']
            elif self.test_type == 'dilated' or self.test_type == 'fcn':
                thisprediction  =  self.model.predict({'input': BatchData},batch_size = batch_size)           
            else:
                raise Exception('Unknown prediction mode {}'.format(self.test_type))
                
            for idx, iden in enumerate(iden_list):
                org_slice = PackList[iden][1]
                extract_slice = PackList[iden][2]
                output[org_slice[0], org_slice[1]] += np.squeeze(thisprediction[idx])[extract_slice[0], extract_slice[1]].copy()
        return output
    def shortCut_FCN(self, inputfile, model=None,windowsize = 1000, board = 40,fixed_window= False,step_size=None,
                     batch_size= 1,**kwargs):
        # receive one image and run the routine job
        if isinstance(inputfile, basestring):
           orgimg = np.asarray(Image.open(inputfile))
        else:
           orgimg = inputfile
        assert self.model or model, 'model not provided!!'
        if  model:
            if self.model != model:
               self.model = model
        #assert len(orgimg.shape) == 3, 'input image must be 3 dimensional at this time'
        
        if len(orgimg.shape) == 2:
           orgimg =  orgimg.reshape(orgimg.shape[0],orgimg.shape[1],1)
        #   orgimg = np.concatenate((orgimg,orgimg,orgimg),axis = 2)
        #orgRowSize , orgColSize = orgimg.shape[0], orgimg.shape[1]
        img = pre_process_img(orgimg, yuv = False)

        BatchData = np.transpose(img, (2,0,1))
        thisprediction = self.get_fcn_split(BatchData, windowsize = windowsize, board = board, batch_size = batch_size,
                        fixed_window= fixed_window,step_size=step_size,**kwargs)
        VotingMap = np.squeeze(thisprediction)
        return VotingMap

    def get_coordinate(self, inputfile, model=None,windowsize = 1000, board = 40,fixed_window= False,step_size=None,
                       probmap=None, threshhold = 0.1, batch_size= 1,min_len=5, **kwargs):
        if probmap is None:
            probmap = self.shortCut_FCN(inputfile = inputfile, model=model,windowsize = windowsize, board = board,
                        batch_size= batch_size, fixed_window= fixed_window,step_size=step_size, **kwargs)
        voting_map = probmap.copy()
        
        voting_map[voting_map < threshhold*np.max(voting_map[:])] = 0
        coordinates = peak_local_max(voting_map, min_distance= min_len, indices = True) # N by 2,
        if coordinates.size == 0:
           coordinates = np.asarray([])           
        return coordinates ,  probmap     

  
    def runtesting(self, **params):
        if self.test_type == 'mdrnn':
            self.folderTesting(**params)
        elif self.test_type == 'fcn' or self.test_type == 'fcn_nosplit' or self.test_type == 'dilated':
            self.folderTesting(**params)
        else:
            self.folderTesting(**params)
            
    def _get_seed_name(self, step, threshhold, min_len, resultmask):
        name  =( resultmask + '_s_' + '{:02d}'.format(step) + '_t_'   + '{:01.02f}'.format(threshhold) \
                 + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
        return name

    def folderTesting(self, **kwargs):
        param = myobj()
        param.windowsize = 500
        param.batch_size = 8
        param.fixed_window = False
        param.step_size = None

        for key in kwargs:
            setattr(param, key, kwargs[key])

        imglist, imagenamelist = getfilelist(self.ImgDir, self.ImgExt)
        #BatchData = np.zeros((self.batchsize, self.patchsize*self.patchsize*self.channel))
        print(self.ImgDir + self.ImgExt[0])
        for imgindx in range(0,len(imglist)):
            print 'processing image {ind}'.format(ind = imgindx)
            if os.path.isfile(imglist[imgindx]):
              orgimg = np.asarray(Image.open(imglist[imgindx]))
              if len(orgimg.shape) == 2:
                 orgimg =  orgimg.reshape(orgimg.shape[0],orgimg.shape[1],1)
                 orgimg = np.concatenate((orgimg,orgimg,orgimg),axis = 2)
            imgname = imagenamelist[imgindx]
            resultDictPath = os.path.join(self.savefolder,  imgname + '_'+ self.resultmask + '.h5')
            resultDictPath_mat = os.path.join(self.savefolder, imgname + '_'+ self.resultmask + '.mat')
            if os.path.isfile(resultDictPath):
               resultsDict = dd.io.load(resultDictPath)
            else:
               resultsDict = {}
            orgRowSize , orgColSize = orgimg.shape[0], orgimg.shape[1]
            for step in self.steppool:
                self.step = step
                print('step is not used in fcn, if you want to use, please modify the folderTesting function. \n')
                #print(self.step)
                votingmapname    = self.resultmask + '_s_' + '{:02d}'.format(self.step) + '_vm'
                voting_time_name = self.resultmask + '_s_' + '{:02d}'.format(self.step) + '_time'
                if self.Probrefresh or votingmapname not in resultsDict.keys():
                   # first pad the image to make it dividable by the labelpatchsize
                    votingStarting_time = time.time()
                    img = pre_process_img(orgimg, yuv = False)

                    VotingMap = np.squeeze(self.get_output(img, windowsize = param.windowsize, batch_size = param.batch_size, 
                                          fixed_window= param.fixed_window, step_size= None))
                    votingEnding_time = time.time()
                    resultsDict[voting_time_name] = votingEnding_time - votingStarting_time
                    print resultsDict[voting_time_name]
                    resultsDict[votingmapname] = np.copy(VotingMap)

                else:
                    VotingMap = resultsDict[votingmapname]
                # display the map if you want
                if self.showmap:
                  plt.imshow(VotingMap, cmap = 'hot')
                  plt.show()
                for threshhold  in self.thresh_pool:
                    for min_len in self.lenpool:
                        thisStart = time.time()
                        localseedname = self._get_seed_name(self.step, threshhold, min_len, self.resultmask)
                       
                        localseedtime = self._get_seed_name(self.step, threshhold, min_len, self.resultmask) + '_time'
                                    
                        if self.Seedrefresh or localseedname not in resultsDict.keys():
                           VotingMap[VotingMap < threshhold*np.max(VotingMap[:])] = 0

                           coordinates = peak_local_max(VotingMap, min_distance= min_len, indices = True) # N by 2,
                           
                           #aa = extrema(VotingMap, kernel_size=(min_len,min_len))
                           
                           if coordinates.size == 0:
                               coordinates = np.asarray([])
                               print("you have empty coordinates for img:{s}".format(s=imgname))
                           thisEnd = time.time()
                           resultsDict[localseedname] = coordinates
                           resultsDict[localseedtime] = thisEnd - thisStart +  resultsDict[voting_time_name]
                           if self.showseed:
                              if coordinates.size > 0:
                                 plt.figure('showseeds')
                                 plt.imshow(orgimg)
                                 plt.plot(coordinates[:,1], coordinates[:,0], 'r.')
                                 plt.show()
                            
            dd.io.save(resultDictPath, resultsDict, compression=None)    #compression='zlib'
            sio.savemat(resultDictPath_mat, resultsDict)
    
    #testingFCN = self.folderTesting
