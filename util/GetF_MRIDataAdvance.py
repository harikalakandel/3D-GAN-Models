
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:55:04 2022

@author: Harikala

'''

https://stackoverflow.com/questions/64674612/how-to-resize-a-nifti-nii-gz-medical-image-file


import skimage.transform as skTrans
im = nib.load(file_path).get_fdata()
result1 = skTrans.resize(im, (160,160,130), order=1, preserve_range=True)



https://datascience.stackexchange.com/questions/90739/how-can-i-create-nii-nifti-file-from-3d-numpy-array

import nibabel as nb

ni_img = nib.Nifti1Image(numpy_array, affine=np.eye(4))
nib.save(ni_img, "dicom_volume_image.nii")



'''
"""

import os
import zipfile
import numpy as np
import glob
import nibabel as nib
from PIL import Image as im
import matplotlib.pyplot as plt
from torchvision.io import read_image
from nilearn.image import resample_img
import sys

import random

from random import shuffle
import cv2

from pathlib import Path
import torch

import skimage.transform as skTrans

from torch import nn




def load_HCPDataset_3_1CH(workingFrom='UNI',folders=None,rFMRIrestList= None,normalize=False,restingState=None):
   
    sys.stdout.flush()
    myData = None
    myAnswerT1 = None
    myAnswerT2 = None
    
    
    if rFMRIrestList == None:    
        rFMRIrestList = ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']#for 7T rfMRI sessions
    #rFMRIrestList = ['rfMRI_REST1_LR','rfMRI_REST1_RL']# for 3T rfMRI sessions
        
    if restingState==None:
        restingState='RestingState-7T'
        
        
    
    
    #rFMRIrestList = ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']
    
    #elif workingFrom == 'UNI':
     #   dataPathX='NSD/nsddata/ppdata/subj0'+sId+'/func1mm/*.nii.gz'
    if workingFrom == 'PC' and folders==None:
        #folders = os.listdir('E:/fMRI-Dataset/RestingState-7T/R_fMRI_Preprocessed')
        folders = ['100610']#,'102311','102816','104416','105923','108323','109123','111312','111514','114823']
    elif folders==None:
        folders = os.listdir('HCP/RestingState-7T/R_fMRI_Preprocessed')  
        #folders = ['100610','102311','102816','104416','105923','108323','109123','111514','114823']
        '''
        folders= ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
                  '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
                  '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
                  '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
                  '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542',
                  '177140','177645','178142','178243','178647','180533','181232','182436','182739','186949',
                  '187345','191033','191336','191841','192641','200311','201515','203418','209228','214019',
                  '214524','221319','239136','246133','249947','257845','263436']
        '''
        
        #folders = ['100610']
    sCount=0
    for cSubject in folders:
        for rest in rFMRIrestList:
              if workingFrom == 'PC':
                  if restingState=='RestingState-7T':
                      currFilePathOne = 'E:/fMRI-Dataset/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseOne_gdc_dc.nii.gz'
                      currFilePathTwo = 'E:/fMRI-Dataset/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseTwo_gdc_dc.nii.gz'
                  else:
                      currFilePathOne = 'E:/fMRI-Dataset/'+restingState+'/'+cSubject+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseOne_gdc_dc.nii.gz'
              
                      currFilePathTwo = 'E:/fMRI-Dataset/'+restingState+'/'+cSubject+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseTwo_gdc_dc.nii.gz'
              elif workingFrom == 'PC_HCP':
                 
                    if restingState=='RestingState-7T':
                        currFilePathOne = 'H:/testInGPU_Cluster/HCP/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseOne_gdc_dc.nii.gz'
                        currFilePathTwo = 'H:/testInGPU_Cluster/HCP/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseTwo_gdc_dc.nii.gz'
                    else:
                        currFilePathOne = 'H:/testInGPU_Cluster/HCP/RestingState_3T/rfMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseOne_gdc_dc.nii.gz'
                        currFilePathTwo = 'H:/testInGPU_Cluster/HCP/RestingState_3T/rfMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseTwo_gdc_dc.nii.gz'
                  
              else:
                  if restingState=='RestingState-7T':
                      currFilePathOne = 'HCP/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseOne_gdc_dc.nii.gz'
                      currFilePathTwo = 'HCP/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseTwo_gdc_dc.nii.gz'
                  else:
                      currFilePathOne = 'HCP/RestingState_3T/rfMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseOne_gdc_dc.nii.gz'
                      currFilePathTwo = 'HCP/RestingState_3T/rfMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseTwo_gdc_dc.nii.gz'
                   
                  
                  
              print(currFilePathOne)
              ff = glob.glob(currFilePathOne)
              test_load=nib.load(ff[0])
              
              
              tmpDataPh1 = test_load.dataobj[:,:,:,:]
              
              #normalize the data
              if normalize:
                  min_D=tmpDataPh1.min(axis=(0,1,2),keepdims=True)
                  max_D=tmpDataPh1.max(axis=(0,1,2),keepdims=True)
                  tmpDataPh1 = (tmpDataPh1 - min_D)/(max_D-min_D)
                  
              
              tmpDataPh1 = np.rollaxis(tmpDataPh1,3,0)             
              tmpDataPh1=np.expand_dims(tmpDataPh1,axis=0)
             
              
            
              
              
              
              
              #combining all fMRI data from each subject
              if not(myData is None):
                myData=np.concatenate((myData,tmpDataPh1),axis=0)
              else:
                myData = tmpDataPh1[:,:,:,:,:]    
                
                
                
              #addding PhaseTwo
              
              ff = glob.glob(currFilePathTwo)
              test_load=nib.load(ff[0])
              
              
                  
              
              
              tmpDataPh2 = test_load.dataobj[:,:,:,:]
              
              #normalize the data
              if normalize:
                  min_D=tmpDataPh2.min(axis=(0,1,2),keepdims=True)
                  max_D=tmpDataPh2.max(axis=(0,1,2),keepdims=True)
                  tmpDataPh2 = (tmpDataPh2 - min_D)/(max_D-min_D)
              
             
              tmpDataPh2 = np.rollaxis(tmpDataPh2,3,0)            
              
              tmpDataPh2=np.expand_dims(tmpDataPh2,axis=0)
              
              #combining all fMRI data from each subject
              myData=np.concatenate((myData,tmpDataPh2),axis=0) 
                
             
        if workingFrom == 'PC':
             currFilePathT1 = 'E:/fMRI-Dataset/Structural-7T/'+cSubject+'/MNINonLinear/T1w_restore.1.60.nii.gz'
             currFilePathT2 = 'E:/fMRI-Dataset/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
        elif workingFrom == 'PC_HCP':
             currFilePathT1 = 'H:/testInGPU_Cluster/HCP/Structural-7T/'+cSubject+'/MNINonLinear/T1w_restore.1.60.nii.gz'
             currFilePathT2 = 'H:/testInGPU_Cluster/HCP/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
        
        else:
             currFilePathT1 = 'HCP/Structural-7T/'+cSubject+'/MNINonLinear/T1w_restore.1.60.nii.gz'
             currFilePathT2 = 'HCP/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
              
        
        
        
        ff = glob.glob(currFilePathT1)
        test_load=nib.load(ff[0])
                
        tmpDataA=test_load.dataobj[:,:,:]
        
        #normalize the data
        if normalize:
            min_D=tmpDataA.min(axis=(0,1,2),keepdims=True)
            max_D=tmpDataA.max(axis=(0,1,2),keepdims=True)
            tmpDataA = (tmpDataA - min_D)/(max_D-min_D)
            
            
        
        
       
        ## in case we remove channels we need to add one dimension for it
        tmpDataA=np.expand_dims(tmpDataA,axis=0)
        #add one more dimension for number of samples..
        tmpDataA=np.expand_dims(tmpDataA,axis=0)
        
              
        
        #combining all fMRI data from each subject
        if not(myAnswerT1 is None):         
          myAnswerT1=np.concatenate((myAnswerT1,tmpDataA),axis=0)          
        else:         
          myAnswerT1 = tmpDataA[:,:,:,:,:]
          
        #adding T2
        
        ff = glob.glob(currFilePathT2)
        test_load=nib.load(ff[0])
        print(test_load.shape)
        
        
        #remove the channel
        tmpDataAT2 = test_load.dataobj[:,:,:]
        #normalize the data
        
        if normalize:
            min_D=tmpDataAT2.min(axis=(0,1,2),keepdims=True)
            max_D=tmpDataAT2.max(axis=(0,1,2),keepdims=True)
            tmpDataAT2 = (tmpDataAT2 - min_D)/(max_D-min_D)
            
        ## in case we remove channels we need to add one dimension for it
        tmpDataAT2=np.expand_dims(tmpDataAT2,axis=0)
        #add one more dimension for number of samples..
        tmpDataAT2=np.expand_dims(tmpDataAT2,axis=0)
            
        
        
        
        if not(myAnswerT2 is None):         
          myAnswerT2=np.concatenate((myAnswerT2,tmpDataAT2),axis=0)          
        else:         
          myAnswerT2 = tmpDataAT2[:,:,:,:,:]
              
       
    return myData,myAnswerT1,myAnswerT2


# ======================================
####################################################################################################

