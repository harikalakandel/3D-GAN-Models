# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:01:53 2022

@author: Harikala
"""
import torch

class HCPDatasetSubjectV1():
    def __init__(self, a,b,c, transform=None):
        #self.df = df
        self.answerT1=b
        self.myData = a
        self.answerT2=c
        #print(len(a))
        self.transform = transform
        
    def __len__(self):
        
        return len(self.answerT1)
    
    
       
    
    def __getitem__(self, subID):
        
        
        startTrain = subID*8
        endTrain = startTrain+6
        trainFMRI = torch.from_numpy(self.myData[startTrain:endTrain,:,:,:,:]).type(torch.float32)
        
        #print('Shape of Train FRMIs ',trainFMRI.shape)
        
        valFMRI = torch.from_numpy(self.myData[endTrain:endTrain+2,:,:,:,:]).type(torch.float32)
        
        mri = torch.from_numpy(self.answerT1[subID]).type(torch.float32)
       
            
        return trainFMRI,valFMRI, mri