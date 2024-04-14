# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:01:53 2022

@author: Harikala
"""
import torch

class HCPDataset():
    def __init__(self, a,b,c, transform=None):
        #self.df = df
        self.answerT1=b
        self.myData = a
        self.answerT2=c
        #print(len(a))
        self.transform = transform
        
    def __len__(self):
        #return len(self.df)
        return len(self.myData)
    
    
       
    
    def __getitem__(self, idx):
       
       
        fMRI = torch.from_numpy(self.myData[idx]).type(torch.float32)
        
        idx=idx//8
        #only consider answerT1
        #now consider same mri from T1 for both 4 + 4 phase one and phase two fmri
        if self.answerT2 == None:
            mri = torch.from_numpy(self.answerT1[idx]).type(torch.float32)
        else:
            #there are four fMRI images of each T1 and T2
            # 1 fMRI -> 1 T1
            #2 fMRI -> 1 T2
            #3 fMRI -> 1 T1
            #4 fMRI -> 1 T2
            #and  so on till 8 fMRI -> 1 T2 then 9 fMRI -> 2 T1  ... 16 fMRI -> 2 T2  
            # so, fMRI to index in T1 or T2 is given by fMRI_index//8
            # fMRI  to either T1 or T2 is defined by odd or even fMRI_index, odd -> T1 even -> T2            
            #print('Current MRI index :',idx)
            if idx % 2 ==0:
                print('T1')
                #4 phase one 
                mri = torch.from_numpy(self.answerT1[idx]).type(torch.float32)
            else:
                print('T2')
                #4 phase two
                mri = torch.from_numpy(self.answerT2[idx]).type(torch.float32)
            
        return fMRI, mri