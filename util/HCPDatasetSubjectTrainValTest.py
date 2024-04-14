# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:01:53 2022

@author: Harikala
"""
import torch

class HCPDatasetSubjectTrainValTest():
    def __init__(self, a,b,c, transform=None,typeSplit='ALL'):
        #self.df = df
        self.answerT1=b
        self.myData = a
        self.answerT2=c
        #print(len(a))
        self.transform = transform
        self.typeSplit=typeSplit
        
    def __len__(self):
        #return len(self.df)
        return len(self.answerT1)
    
    
       
    
    def __getitem__(self, subID):
        #print('Current SubID is ',subID)
        
        
        if self.typeSplit=='ALL':
            startTrain = subID*8
            endTrain = startTrain+4   #train 50  4 out of 8
            trainFMRI = torch.from_numpy(self.myData[startTrain:endTrain,:,:,:,:]).type(torch.float32)
            
            #print('Shape of Train FRMIs ',trainFMRI.shape)
            #validate 25 2 out of 8
            valFMRI = torch.from_numpy(self.myData[endTrain:endTrain+2,:,:,:,:]).type(torch.float32)
            #test 25 2 out of 8
            testFMRI = torch.from_numpy(self.myData[endTrain+2:endTrain+4,:,:,:,:]).type(torch.float32)
        elif self.typeSplit=='TrainValidate':
            startTrain = subID*8
            endTrain = startTrain+6   #train 75  6 out of 8
            trainFMRI = torch.from_numpy(self.myData[startTrain:endTrain,:,:,:,:]).type(torch.float32)
            
            #print('Shape of Train FRMIs ',trainFMRI.shape)
            #validate 25 2 out of 8
            valFMRI = torch.from_numpy(self.myData[endTrain:endTrain+2,:,:,:,:]).type(torch.float32)
            #test 25 2 out of 8
            testFMRI = []
        
        mri = torch.from_numpy(self.answerT1[subID]).type(torch.float32)
        
        #print('Original shape of mri :: ',mri.shape)
        #print('Original shape of trainFMRI :: ',trainFMRI.shape)
       
            
        return trainFMRI,valFMRI,testFMRI, mri