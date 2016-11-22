from __future__ import absolute_import
import numpy as np
import os
from local_utils import*
from ImageGenerator import *

from scipy.io import loadmat
from PIL import Image, ImageDraw
#from skimage.color import rgb2gray
import skimage, skimage.morphology
import math
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from  scipy.ndimage.interpolation import rotate
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from BaseExtractor import BaseExtractor
import json
#from recurrentUtils import * 
#import matplotlib.image as mpimgs

class FcnExtractor(BaseExtractor):
    def __init__(self, initial_data):
        self.usecontour = 0
        self.dilate = None
        self.erode = None
        #self.usegrouping = 0
        self.usingindx = 1
        self.usegray = 0
        self.padsize = 0           #floor((fineTsize + labelpatchsize)/2);
        self.pickratio = 0.9       # 1 means take all the islet pixels
        self.negratio = 0.99       # the ratio of the number of negtive pts you wanna choose from.
        self.BalenceRatio = 1
        self.patchPerbatch = 1E+5
        self.extendratio =  [1.2]
        self.resizeratio = [1]
        self.channel = 3
        self.label_channel = 2
        self.pick_band = None
        self.rotatepool  = [0,45,90,135, 180, 225, 270]
        self.use_weighted = False
        self.reverse_channel = False
        self.filp_channel = False
        #self.mask_thresh = 10
        #self.mask_prob = 0.1
        self.w = 5
        self.wc = 1
        self.double_output = False
        
        self.decayparam = {}
        self.decayparam['alpha'] = 1
        self.decayparam['r'] = 4
        self.decayparam['scale'] = 1
        self.decayparam['decalratio'] = 1
        self.decayparam['smallweight'] = 0.05
        self.decayparam['use_gauss'] = 1

        #self.mode  = 'zigzag'
        self.maximg = float('nan')

        self.selected_portion = 1
        self.selected_num = 1
        self.mask_dilate = 0
        self.period  = 1
        self.random_pick = True
        self.ImageGenerator_Identifier = 'process_cell_mask'
        self.boarder = 0
        
        self.get_validation = False
        
        self.datainfo = None
        for key in initial_data:
            setattr(self, key,initial_data[key])
        super(FcnExtractor, self).__init__(initial_data)

    def getMatinfo_volume(self):
        ImgContainer = {}
        imgIndx = []
        AllIndx = []
        realImgindx = -1
        ImageGenerator = self.getImageGenerator()

        for imgindx, outputs_dict in enumerate(ImageGenerator(self)):
            
            img = outputs_dict['img']
            mask_img = outputs_dict['mask']
            filled_img = outputs_dict['filled_img']
            shed_mask = outputs_dict['shed_mask']

            img = self.image_prep(img)
            mask_img = self.standardize(mask_img)
            filled_img = self.standardize(filled_img)
          
            npad3 = ((self.padsize,self.padsize),(self.padsize,self.padsize),(0,0))
            #npad2 = ((self.padsize,self.padsize),(self.padsize,self.padsize))
            img = np.pad(img, npad3, 'symmetric')
            #mask_img = np.pad(mask_img,npad3, 'symmetric')
            filled_img = np.pad(filled_img,npad3, 'symmetric')
            row_size, col_size = filled_img.shape[0:2]

            if not self.patchsize: #if patchsize is none, means we take the whole image
                ThisSeletedInd = list(np.ravel_multi_index((img.shape[0]/2, img.shape[1]/2), dims=(img.shape[0], img.shape[1])))
            else:  # if the patchsize is fixed
                if self.random_pick == True:
                    restricted_region = np.ones(filled_img.shape)
                
                    if self.pick_band is not None:
                        # this is supposed to restrict the random region 
                        if 0< self.pick_band[0] < 1:
                            #means we use ratio to specify
                            row_band = max(1, self.pick_band[0]*row_size)
                            col_band = max(1, self.pick_band[1]*col_size)
                        else:
                            row_band, col_band = self.pick_band     
                        restricted_region = restricted_region * np.nan
                        se = CentralToOrigin(row_size/2, col_size/2, row_band, col_band)
                        restricted_region[se['RS'] : se['RE'], se['CS']:se['CE']] = 1
                    else:
                        
                        right_row = max( (row_size -self.patchsize + 10 ), min(5, row_size) )
                        right_col = max( (col_size -self.patchsize + 10 ), min(5, col_size) )
                        se = CentralToOrigin(row_size/2, col_size/2, right_row, right_col)
                        restricted_region = restricted_region * np.nan
                        restricted_region[se['RS'] : se['RE'], se['CS']:se['CE']] = 1

                    allcandidates = shuffle(find(restricted_region == 1)) 
                    total_num = len(allcandidates)
                    selected_num = min(self.maxpatch, int(math.ceil(self.pickratio * total_num)) )
                    
                    ThisSeletedInd  = allcandidates[0:selected_num]                 
                else:
                    # if you want some balance between positive and negtive
                    posind = find((filled_img == 1) | (filled_img == 255) ) 
                    posind, randindx = shuffle(posind, np.arange(0,len(posind)),random_state = 1)
                    realPosind = posind[np.arange(int(math.ceil(self.pickratio *len(posind)) ))    ]
                    realNumPos = len(realPosind)
                    otherInd = find(~((filled_img == 1) | (filled_img == 255)))
                    otherInd = shuffle(otherInd,random_state = 1)
                    ThisOtherInd = []
                    if self.usegrouping == 1:
                        curRandOtherIndx = otherInd
                        # cluster the data to k groups
                        thislenNeg = len(curRandOtherIndx)
                        if testAllTrainingData.shape[0] <= thislenNeg:
                            testAllTrainingData = np.zeros(thislenNeg, self.patchsize *self.patchsize * self.channel)
                        NumberGroups = 64
                        self.__Points2Patches(testAllTrainingData[0:thislenNeg,:],curRandOtherIndx, img, [self.patchsize, self.patchsize])
                        # this modify testAllTrainingData in those function
                        mbk = MiniBatchKMeans(init='k-means++', n_clusters=NumberGroups, batch_size=10000,
                            n_init=10, max_no_improvement=10, verbose=0)
                        mbk.fit(testAllTrainingData[0:thislenNeg,:])
                        groupID = mbk.labels_
                        uniqueGroup = np.unique(groupID)
                        NumberGroup = []
                        for uniIdx in range(len(uniqueGroup)):
                            thisgroup = uniqueGroup[uniIdx]
                            thisIdind = curRandOtherIndx[groupID == thisgroup]
                            thisNumCount = len(thisIdind)
                            NumberGroup.append(thisNumCount)

                        meannum = (NumberGroups/30)*np.median(np.asarray(NumberGroup))

                        for uniIdx in range(len(uniqueGroup)):
                            thisgroup = uniqueGroup[uniIdx]
                            thisIdind = curRandOtherIndx[groupID == thisgroup]
                            thisNumCount = len(thisIdind)

                            thisIdind = shuffle(thisIdind,random_state = 1)
                            selectInd = thisIdind[1: min(meannum,thisNumCount)];
                            ThisOtherInd += list(selectInd)
                    else:
                        numberSelectNeg = min(len(otherInd), math.floor(self.BalenceRatio* realNumPos ))  ;
                        choosenOtherInd = otherInd[ 0: numberSelectNeg];
                        ThisOtherInd +=  list(choosenOtherInd)
                    #choosenOtherInd = ThisOtherInd[1:  min(len(ThisOtherInd),math.floor(self.BalenceRatio*realNumPos )  )];
                    ThisSeletedInd  = list(realPosind)+list(ThisOtherInd)
                    #mask_img_rotation.flat[np.asarray(ThisSeletedInd)] = 1
            realImgindx += 1
            ImgContainer[(realImgindx,0)] = img #rotate(img, rotate_id)
            ImgContainer[(realImgindx,1)] = mask_img #rotate(mask_img, rotate_id)
            if self.double_output:
                ImgContainer[(realImgindx,2)] = shed_mask #rotate(mask_img, rotate_id)
            AllIndx += list(ThisSeletedInd)
            imgIndx += list(np.tile(realImgindx, len(ThisSeletedInd)))

        AllIndx, imgIndx = shuffle(AllIndx, imgIndx)
        selectnum = min(len(AllIndx), self.maxsamples)
        AllIndx = AllIndx[:selectnum]
        imgIndx = imgIndx[:selectnum]
        datainfo = {}
        datainfo['outputdim'] = self.labelpatchsize *self.labelpatchsize *self.label_channel
        datainfo['inputdim'] = self.patchsize *self.patchsize *self.channel
        datainfo['h'] = self.patchsize
        datainfo['w'] = self.patchsize
        datainfo['channel'] = self.channel
        datainfo['Totalnum'] = len(imgIndx)
        self.TrainingMatinfo['datainfo'] = datainfo
        self.TrainingMatinfo['ImgContainer'] = ImgContainer
        self.TrainingMatinfo['AllIndx'] = np.asarray(AllIndx)
        self.TrainingMatinfo['imgIndx'] = np.asarray(imgIndx)
        return self.TrainingMatinfo


class ListExtractor(BaseExtractor):
    def __init__(self, initial_data):
        self.nameList = []  # should only contains image name with extension
        self.labelList = [] # should be one-hot coding or simply a list of int
        self.local_norm = True
        self.destin_shape = None
        self.datainfo = None
        self.annodir = None
        self.annoExt = '.json'
        self.anno_count = 0
        for key in initial_data:
            setattr(self, key,initial_data[key])
        
        self.nb_sample = len(self.labelList)   
        if type(self.labelList[0]) is list:
            self.nb_class = len(self.labelList[0])
        else:
            self.nb_class = max(self.labelList)
        
        super(ListExtractor, self).__init__(initial_data)
        self.getMatinfo()

    def _standardize_label(self, label):
        if type(label) in [int, float]:
            label = to_one_hot(int(label), self.nb_class)
            
        elif type(label) in [list, tuple]:
            label = np.asarray(label)
        
        if type(label) is not np.ndarray:
            raise Exception('Wrong label input as : {s}'.format(s =  str(type(label))) )   
        return label
    
    def getOneDataBatch_stru(self, thisRandIndx=None, thisbatch=None, thislabel=None):
        if thisRandIndx is None:
            thisRandIndx = np.arange(0, len(self.nameList))
        if thisbatch is None:
            if self.datainfo is None:
                self.getMatinfo()
            thisbatch = np.zeros((self.datainfo['Totalnum'],) + self.datainfo['inputshape']) 
        if thislabel is None:
            thislabel = np.zeros((self.datainfo['Totalnum'],) + self.datainfo['outputshape']) 
        thisnum = 0
        for ind in thisRandIndx:
            thisname = self.nameList[ind]
            thislabel_list = self.labelList[ind]

            valid = False
            for imgExt in self.dataExt:
                thisfile = thisname + imgExt
                thispath = os.path.join(self.datadir, thisfile)
                if os.path.isfile(thispath):
                   img = imread(thispath)
                   valid = True
                   break
            if valid:
          
                img  = pre_process_img(img, yuv = False,norm=self.local_norm)
                if self.destin_shape is not None:
                    shape = tuple(self.destin_shape) + (img.shape[2],)
                    img =  imresize_shape(img,shape)
                # transpose the img to order (channel, row, col)
                img = np.transpose(img, (2,0,1))
                thisbatch[thisnum,...] = img
                thislabel[thisnum,...] = self._standardize_label(thislabel_list)
                thisnum += 1
            else:
                print('Image: {s} not find'.format(s = thisname))
        return thisbatch, thislabel

    
        
    def getImg_Anno(self,thisRandIndx=None, thisbatch=None, thisanno=None):
        if thisRandIndx is None:
            thisRandIndx = np.arange(0, len(self.nameList))
        if thisbatch is None:
            if self.datainfo is None:
                self.getMatinfo()
            thisbatch = np.zeros((self.datainfo['Totalnum'],) + self.datainfo['inputshape']) 
        if thisanno is None:
            thisanno = [None for _ in range(self.datainfo['Totalnum'])]
        thisnum = 0
        for ind in thisRandIndx:
            thisname = self.nameList[ind]
            valid = False
            for imgExt in self.dataExt:
                thisfile = thisname + imgExt
                thispath = os.path.join(self.datadir, thisfile)
                img = imread(thispath)
                if os.path.isfile(thispath):
                   valid = True
                   break
            if valid: 
                         
                if self.destin_shape is not None:
                    shape = (img.shape[2],) + tuple(self.destin_shape) 
                else:
                    shape = None
                img = get_cnn_img(thispath,shape, self.local_norm)  

                thisbatch[thisnum,...] = img
                cap_tuple = (self._get_parse_anno(thisname), self.anno_count, thisname)
                self.anno_count += 1
                thisanno[thisnum]= cap_tuple
                thisnum += 1
            else:
                print('Image: {s} not find'.format(s = thisname))
        return thisbatch, thisanno
    def _get_parse_anno(self, thisname):
        thisfile = thisname + self.annoExt
        thispath = os.path.join(self.annodir, thisfile)
        with open(thispath) as data_file:    
            anno_dict = json.load(data_file)
            thisstr = ''
            for k, v in anno_dict.iteritems():
                thisstr = thisstr + ' '+ k.title() +': '
                thisstr  = thisstr + ' ' + v[1]
        return thisstr
    def getMatinfo(self):
        datainfo = {}
        ind = 0
        thisname = self.nameList[ind]
        #thislabel = self.labelList[ind]
        for imgExt in self.dataExt:
                thisfile = thisname + imgExt
                thispath = os.path.join(self.datadir, thisfile)
                img = imread(thispath)
                if os.path.isfile(thispath):
                    if self.destin_shape is not None:
                        shape = (img.shape[2],) + tuple(self.destin_shape) 
                    else:
                        shape = None
                    img = get_cnn_img(thispath,shape, self.local_norm)  

        datainfo['outputdim'] = self.nb_class
        datainfo['inputdim'] =  np.prod(img.shape)
        datainfo['outputshape'] = (self.nb_class,)
        datainfo['inputshape'] =  img.shape

        datainfo['h'] = img.shape[1]
        datainfo['w'] = img.shape[2]
        datainfo['channel'] = img.shape[0]
        datainfo['Totalnum'] = len(self.nameList)
        self.datainfo = datainfo
        self.TrainingMatinfo['datainfo'] = datainfo
        return self.TrainingMatinfo    

def get_cnn_img(thispath, shape, local_norm):
    '''
    shape: should be 3*224*224 pr None

    '''
    img = imread(thispath)
    img  = pre_process_img(img, yuv = False,norm= local_norm)
    if shape is not None:
        img =  imresize_shape(img,(shape[1], shape[2], shape[0]))
    # transpose the img to order (channel, row, col)
    img = np.transpose(img, (2,0,1))
    return img