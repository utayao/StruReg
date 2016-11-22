from __future__ import absolute_import
import numpy as np
import os
from scipy.io import loadmat

#from skimage.color import rgb2gray
import skimage, skimage.morphology
import math
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from  scipy.ndimage.interpolation import rotate
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from local_utils import*
from ImageGenerator import *

#from recurrentUtils import *
#import matplotlib.image as mpimgs

class BaseExtractor(object):
    def __init__(self, initial_data):
        self.datadir = ""
        self.maxsamples = 1.5E+6
        self.contourext = ["_withcontour"]
        self.decalratio = 0.5
        self.labelExt = '.mat'
        self.dataExt =    [".jpg",'.tif']
        self.patchsize   = 40
        self.labelpatchsize   = 40
        self.contourname   = 'Contours'
        #self.usecontour = 0;
        #self.dialate = 0;
        #self.usegrouping = 0;
        #self.usingindx = 1
        #self.usegray = 0
        #self.padsize = 0;           #floor((fineTsize + labelpatchsize)/2);
        #self.pickratio = 0.9;       # 1 means take all the islet pixels
        #self.negratio = 0.99;       # the ratio of the number of negtive pts you wanna choose from.
        #self.BalenceRatio = 1;
        #self.patchPerbatch = 1E+5;
        #self.extendratio =  [1.2];
        #self.resizeratio = [1];
        #self.channel = 3;
        #self.rotatepool  = [0,45,90,135, 180, 225, 270]
        self.TrainingMatinfo   = {}
        self.meanstdstatus = 0
        self.thismean = []
        self.thisdev = []
        self.firstcall = True

        #self.mask_thresh = 10
        #self.mask_prob = 0.1
        #self.decayparam = {}
        #self.decayparam['alpha'] = 1
        #self.decayparam['r'] = 4
        #self.decayparam['scale'] = 1

        #self.mode  = 'zigzag'
        #self.maximg = float('nan')
        self.ImageGenerator = None
        self.crop_patch_size = None
        #self.selected_portion = 1
        #self.selected_num = 1
        #self.mask_dilate = 0
        #self.period  = 100
        #self.random_pick = True
        #self.ImageGenerator_Identifier = 'process_cell_mask'

        for key in initial_data:
            setattr(self, key,initial_data[key])
    def rgb2gray(self,rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    def getdatainfo(self):
        #datainfo['featuredim'] = self.labelpatchsize *self.labelpatchsize
        #datainfo['inputdim'] = self.patchsize *self.patchsize *3
        #datainfo['h'] = self.patchsize
        #datainfo['w'] = self.patchsize
        #datainfo['channel'] = self.channel
        #datainfo['Totalnum'] = len(imgIndx)
        return self.TrainingMatinfo['datainfo']

    def setMatinfo(self, Matinfo):
        self.TrainingMatinfo = Matinfo
        self.firstcall = False

    def standardize(self, img):
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0],img.shape[1],1)
        return img

    def image_prep(self, img):
        if len(img.shape) == 2:
            if self.channel == 3:
                img =  img.reshape(img.shape[0],img.shape[1],1)
                img = np.concatenate((img,img,img),axis = 2)
            else:
                img =  img.reshape(img.shape[0],img.shape[1],1)
        else:
            if self.channel == 1:
                if img.shape[2] == 3:
                    img =  self.rgb2gray(img)
                elif img.shape[2] == 1:
                    pass
                else:
                    raise Exception('Incorrect image shape[3] and channel setting!')
        return img
        
    def calMeanDevPts(self):
        if not self.TrainingMatinfo:
            print "Empty TrainingMatinfo,check if you have call getMatinfo first"
            return 
        datainfo = self.TrainingMatinfo['datainfo']
        inputdim = datainfo['inputdim']
        Totalnum = datainfo['Totalnum']
        numberofmat = int((Totalnum + self.patchPerbatch - 1)//self.patchPerbatch)
        #train_x_tmp = np.zeros([self.patchPerbatch, inputdim])
        randindex_ = shuffle(np.arange(Totalnum)) # ordered index of the total number
        startCount = 0
        thissum = 0
        train_x_tmp = np.zeros((self.patchPerbatch, inputdim))
        for matidx in range(numberofmat):
            reminingnum = Totalnum - matidx*self.patchPerbatch
            #if self.patchPerbatch * (matidx + 1) > Totalnum:
            #    train_x_tmp = np.zeros(reminingnum, inputdim)
            numbercase = int( min(reminingnum,self.patchPerbatch))
            this_rand_indx = randindex_[np.arange(numbercase) + startCount]
            if self.usingindx == 1:
                self.getOneDataBatch(this_rand_indx, train_x_tmp[0:numbercase,:])
            else:
                train_x_tmp[0:numbercase,:] = self.TrainingMatinfo['AllTrainingData'][this_rand_indx,:]
            startCount += numbercase
            thissum += np.sum(train_x_tmp[0:numbercase,:],axis =0)
        thismean = thissum/Totalnum
        #-----------Deal with the deviation----------------------------
        DeviSum = 0
        startCount = 0
        #train_x_tmp = np.zeros(self.patchPerbatch, inputdim)
        for matidx in range(numberofmat):
            reminingnum = Totalnum - matidx*self.patchPerbatch
            #if self.patchPerbatch * (matidx + 1) > Totalnum:
            #       train_x_tmp = np.zeros(reminingnum, inputdim)
            numbercase = int(min(reminingnum,self.patchPerbatch))
            this_rand_indx = randindex_[np.arange(numbercase)  + startCount]
            if self.usingindx == 1:
                self.getOneDataBatch(this_rand_indx, train_x_tmp[0:numbercase,:])
            else:
                train_x_tmp[0:numbercase,:] = self.TrainingMatinfo['AllTrainingData'][this_rand_indx,:]
            startCount += numbercase
            MediumT = train_x_tmp[0:numbercase,:] - thismean[np.newaxis,:]
            DeviSum += np.sum(MediumT * MediumT,axis =0)
        thisdev = np.sqrt(DeviSum/Totalnum)
        self.meanstdstatus = 1
        self.thismean = thismean
        self.thisdev = thisdev
        return thismean, thisdev
       
    def getOneDataBatch_stru(self,thisRandIndx=None, thisbatch=None, thislabel=None, secondlabel = None):
        """
        thisRandIndx, is just a sclice of the index of the total index of ordered indx : 1:1536000
        thisbatch is an numpy array, passed as reference to speed up.
        datainfo['h'] = self.patchsize
        datainfo['w'] = self.patchsize
        datainfo['channel'] = self.channel
        datainfo['Totalnum'] = len(imgIndx)

        self.TrainingMatinfo['datainfo'] = datainfo
        self.TrainingMatinfo['ImgContainer'] = ImgContainer
        self.TrainingMatinfo['AllIndx'] = np.asarray(AllIndx)
        self.TrainingMatinfo['imgIndx'] = np.asarray(imgIndx)  """
        #featuredim = self.datainfo['featuredim']
        #thisbatch = np.zeros(len(thisRandIndx),featuredim)       
        
        orderedInd = np.arange(0,len(thisRandIndx))
        ALlimageGroup = self.TrainingMatinfo['imgIndx'][thisRandIndx] #10,34,21,1,2,2,1
        Allmixindx    = self.TrainingMatinfo['AllIndx'][thisRandIndx]
        uniqueImage,counts_ = np.unique(ALlimageGroup,return_counts=True)
        amount = 0;
        tmpdata = np.zeros((max(counts_), thisbatch.shape[1]))
        tmplabel = np.zeros((max(counts_), thislabel.shape[1]))
        if self.double_output:
            tmpsecondlabel = np.zeros((max(counts_), secondlabel.shape[1]))
        
        for uniIdx in range(len(uniqueImage)):
            thisimg = uniqueImage[uniIdx]
            thisImgind = Allmixindx[ALlimageGroup == thisimg] # it preserve the order
            thisorderIdind = orderedInd[ALlimageGroup == thisimg] #propagate the order to orderedind
            img = self.TrainingMatinfo['ImgContainer'][(thisimg,0)]
            maskimg = self.TrainingMatinfo['ImgContainer'][(thisimg,1)]

            self.__Points2Patches(tmpdata[0:len(thisorderIdind)],thisImgind, img, [self.patchsize, self.patchsize])
            self.__Points2Patches(tmplabel[0:len(thisorderIdind)], thisImgind, maskimg, [self.labelpatchsize, self.labelpatchsize])
            thisbatch[thisorderIdind,:] = tmpdata[0:len(thisorderIdind)]
            thislabel[thisorderIdind,:] = tmplabel[0:len(thisorderIdind)]
            if self.double_output:
                second_img = self.TrainingMatinfo['ImgContainer'][(thisimg,2)]
                self.__Points2Patches(tmpsecondlabel[0:len(thisorderIdind)], thisImgind, second_img, [self.labelpatchsize, self.labelpatchsize])
                secondlabel[thisorderIdind,:] = tmpsecondlabel[0:len(thisorderIdind)]
            
            """Patches should be pre allocated to save computational time"""
            amount += len(thisImgind)
        assert amount == len(thisRandIndx)


    def getOneDataBatch(self,thisRandIndx, thisbatch):
        orderedInd = np.arange(0,len(thisRandIndx))
        ALlimageGroup = self.TrainingMatinfo['imgIndx'][thisRandIndx] #10,34,21,1,2,2,1
        Allmixindx    = self.TrainingMatinfo['AllIndx'][thisRandIndx]
        uniqueImage,counts_ = np.unique(ALlimageGroup,return_counts=True)
        amount = 0
        tmpdata = np.zeros((max(counts_), thisbatch.shape[1]))
        for uniIdx in range(len(uniqueImage)):
            thisimg = uniqueImage[uniIdx]
            thisImgind = Allmixindx[ALlimageGroup == thisimg] # it preserve the order
            thisorderIdind = orderedInd[ALlimageGroup == thisimg] #propagate the order to orderedind
            img = self.TrainingMatinfo['ImgContainer'][(thisimg,0)]
            self.__Points2Patches(tmpdata[0:len(thisorderIdind)],thisImgind, img, [self.patchsize, self.patchsize])
            thisbatch[thisorderIdind,:] = tmpdata[0:len(thisorderIdind)]
            #self.__Points2Patches(thislabel[thisorderIdind,:],thisImgind, maskimg, [self.labelpatchsize, self.labelpatchsize])
            # Patches should be pre allocated to save computational time
            amount += len(thisImgind)
        assert amount == len(thisRandIndx)
      
    def getImageGenerator(self):
        ImageGenerator = None
        if self.ImageGenerator is not None:
           return self.ImageGenerator
        if self.volume == 0 or self.volume == 'image':
            ImageGenerator = yieldImages
        elif self.volume == 1 or self.volume == 'volume':
            ImageGenerator = yieldImagesfromVolume
        elif self.volume == 'paramesium':
            ImageGenerator = yieldImages_paramisum
        elif self.volume == 2 or self.volume == 'mia':
            ImageGenerator = yieldImages
        elif self.volume == 3 or self.volume == 'ImgLabel':
            ImageGenerator = yieldImages_ImgLabel
        return   ImageGenerator
        
    def getMatinfo(self):
        ImgContainer = {}
        imgIndx = []
        AllIndx = []
        realImgindx = -1
        ImageGenerator = self.getImageGenerator();
        for imgindx, pack_ in enumerate(ImageGenerator(self)):
            if imgindx == self.maximg:
               break
            
            img, mask_img, filled_img = pack_
            if np.mod(imgindx,self.period) == 0:
                print "precessing img: {s}".format(s = imgindx)
            if 1:
                img = self.image_prep(img)
                mask_img = self.standardize(mask_img)

                npad3 = ((self.padsize,self.padsize),(self.padsize,self.padsize),(0,0))
                npad2 = ((self.padsize,self.padsize),(self.padsize,self.padsize))
                img = np.pad(img, npad3, 'symmetric')
                mask_img = np.pad(mask_img,npad3, 'symmetric')
                filled_img = np.pad(filled_img,npad2, 'symmetric')
                
                if not self.patchsize: #if patchsize is none, means we take the whole image
                    ThisSeletedInd = list(np.ravel_multi_index((img.shape[0]/2, img.shape[1]/2), dims=(img.shape[0], img.shape[1])))
                else:  # if the patchsize is fixed
                        if self.random_pick == True:
                            allcandidates = shuffle(find(mask_img != np.nan))
                            total_num = len(allcandidates)
                            selected_num = min(self.maxpatch, int(self.pickratio *total_num) )
                            ThisSeletedInd  = allcandidates[0:selected_num]
                        else:
                            posind = self.__find(filled_img == 1)
                            posind, randindx = shuffle(posind, np.arange(0,len(posind)),random_state = 1)
                            realPosind = posind[np.arange(int(self.pickratio * len(posind)))    ]
                            realNumPos = len(realPosind)
                            otherInd = self.__find(filled_img == 0)
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
                AllIndx += list(ThisSeletedInd)
                imgIndx += list(np.tile(realImgindx, len(ThisSeletedInd)))

        AllIndx, imgIndx = shuffle(AllIndx, imgIndx,random_state = 1)
        selectnum = min(len(AllIndx), self.maxsamples)
        AllIndx = AllIndx[:selectnum]
        imgIndx = imgIndx[:selectnum]
        datainfo = {}
        datainfo['outputdim'] = self.labelpatchsize *self.labelpatchsize
        datainfo['inputdim'] = self.patchsize *self.patchsize *self.channel
        datainfo['h'] = self.patchsize
        datainfo['w'] = self.patchsize
        datainfo['channel'] = self.channel
        datainfo['Totalnum'] = len(imgIndx)
        self.datainfo = datainfo
        self.TrainingMatinfo['datainfo'] = datainfo
        self.TrainingMatinfo['ImgContainer'] = ImgContainer
        self.TrainingMatinfo['AllIndx'] = np.asarray(AllIndx)
        self.TrainingMatinfo['imgIndx'] = np.asarray(imgIndx)
        return self.TrainingMatinfo


    def Points2Patches(self,centerIndx, img, patchsize):
        totalsub = np.unravel_index(centerIndx, [img.shape[0],img.shape[1]])
        numberofInd = len(centerIndx)
        #
        if len(img.shape) == 2:
            img = img[:,:,None]
        Patches = np.zeros(numberofInd, np.prod(patchsize)*img.shape[2])
        npad3 = ((patchsize[0],patchsize[1]),(patchsize[0],patchsize[1]),(0,0))
        img = np.pad(img,npad3, 'symmetric')
        centralRow = totalsub[0][:] + patchsize[0]
        centralCol = totalsub[1][:] + patchsize[1]

        se = self.__CentralToOrigin(centralRow, centralCol,patchsize[0],patchsize[1])

        for i in range(numberofInd):
            Patches[i,:] = img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:].flatten().copy()
        return Patches
    def __Points2Patches(self, Patches,centerIndx, img, patchsize):
        """Patches should be pre allocated to save computational time, need attention
        to make sure that Patches is changed after calling this function, if pathchsize
        are none, take the whole image"""
        #totalsub = ind2sub([img.shape[0],img.shape[1]], centerIndx)
        if (not patchsize[0]) and (not patchsize[1]):
           patchsize = img.shape[0:2]

        totalsub = np.unravel_index(centerIndx, [img.shape[0],img.shape[1]])
        numberofInd = len(centerIndx)
        #Patches = np.zeros(numberofInd, np.prod(patchsize)*img.shape[2])
        if len(img.shape) == 2:
            img = img[:,:,None]
        npad3 = ((patchsize[0],patchsize[1]),(patchsize[0],patchsize[1]),(0,0))
        img = np.pad(img,npad3, 'symmetric')
        centralRow = totalsub[0][:] + patchsize[0]
        centralCol = totalsub[1][:] + patchsize[1]

        se = self.__CentralToOrigin(centralRow, centralCol,patchsize[0],patchsize[1])

        for i in range(numberofInd):
            Patches[i,:] = img[se['RS'][i] : se['RE'][i], se['CS'][i]:se['CE'][i],:].flatten().copy()
        #return Patches

    def __CentralToOrigin(self, centralRow, centralCol,Rowsize,Colsize):
        RowUp = int(Rowsize/2)
        RowDown = Rowsize - RowUp
        ColLeft = int(Colsize/2)
        ColRight = Colsize - ColLeft
        se = {}
        se['RS'] = centralRow - RowUp
        se['RE'] = centralRow + RowDown
        se['CS'] = centralCol - ColLeft
        se['CE'] = centralCol + ColRight

        return se

    def __find(self, logicalMatrix):
        totalInd = np.arange(0, len(logicalMatrix.flat))
        return totalInd[logicalMatrix.flatten()]
    def __knnsearch(self, seeds, pints,K):
        """return the indexes and distance of k neareast points for every pts in points from seeds\
        seeds: N*dim, points: N*dim
        seeds, and points should be of N*dim format"""
        knn = NearestNeighbors(n_neighbors=K)
        knn.fit(seeds)
        distance, index  = knn.kneighbors(pints, return_distance=True)
        return index,distance

    def __knnsearch_self(self,seeds,points,K):
        """return the indexes and distance of k neareast points for every pts in points from seeds\
        seeds: N*dim, points: N*dim"""
        nseeds = seeds.shape[0]
        npoints = points.shape[0]
        #dim = seeds.shape[1]
        ind_dis = np.zeros([npoints,2,K])
        distmatrix = pairwise_distances(points, seeds,metric = 'euclidean')
        for kidx in range(K):
            for pind in range(0,npoints):
                leastInd = self.__quickSelect(distmatrix[pind,:], range(nseeds),kidx)
                ind_dis[pind,:,kidx]=np.array([leastInd, distmatrix[pind,leastInd]])
        return ind_dis[:,0,:], ind_dis[:,1,:]
    def __quickSelect(self,seq,seqInd,k):
        len_seq = len(seq)
        if len_seq < 2:return seqInd[0]
        ipivot = len_seq // 2
        pivot = seq[ipivot]
        pivotInd = seqInd[ipivot]
        smallerList, smallerInd, largerList, largerInd = [],[],[],[]
        for i,x in enumerate(seq):
            if x<= pivot and i!= ipivot:
                smallerList.append(x)
                smallerInd.append(seqInd[i])
            if x>pivot and i!= ipivot:
                largerList.append(x)
                largerInd.append(seqInd[i])

        m = len(smallerList)
        if k == m:
           return pivotInd
        elif k < m:
           return self.__quickSelect(smallerList,smallerInd,k)
        else:
           return self.__quickSelect(largerList,largerInd,k-m-1)

    def roipoly(self, rowsize,colsize,xcontour,ycontour):
        polyvet = np.zeros([1,2*max(xcontour.shape[0], xcontour.shape[1])])
        polyvet[0::2] = xcontour
        polyvet[1::2] = ycontour
        img = Image.new('L', (colsize, rowsize), 0)
        ImageDraw.Draw(img).polygon(polyvet,outline=1,fill =1)
        return np.array(img)

    def __getfilesinfo(self):
        alllist  = [f for f in os.listdir(self.imgdir)]
        absfilelist = [];
        absmatfilelist = [];
        for f in alllist:
            if os.path.isfile(os.path.join(self.imgdir,f)) and \
                       os.path.splitext(f)[1] in self.ImgExt:
               #if the image file exist
               for contourext in self.contourext:
                 thismatfile  = os.path.join(self.imgdir,os.path.splitext(f)[0] + \
                 contourext + self.LabelExt)
                 if os.path.isfile(thismatfile):
                    absmatfilelist.append(thismatfile)
                    absfilelist.append(os.path.join(self.imgdir,f))
                    break
                 else:
                    print "Image: {s} does not have matfile".format(s = os.path.splitext(f)[0] )
        return absfilelist, absmatfilelist

