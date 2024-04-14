import os
import h5py
from pathlib import Path
from math import exp
import numpy as np
import nibabel as nib
import glob

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid

from matplotlib import pyplot as plt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


def get_gradient_penalty(real_X, fake_X, discriminator,device='cpu'):
    alpha = torch.rand(size=(real_X.size(0), 1, 1, 1, 1), dtype=torch.float32)
    alpha = alpha.repeat(1, *real_X.size()[1:])
    
    alpha=alpha.to(device)

    interpolates = alpha * real_X + ((1 - alpha) * fake_X)
    interpolates = interpolates.requires_grad_(True)
    interpolates = interpolates.to(device)
    output = discriminator(interpolates)

    ones = torch.ones(size=output.size(), dtype=torch.float32)
    ones=ones.to(device)

    gradients = torch.autograd.grad(outputs=output, inputs=interpolates, grad_outputs=ones, create_graph=True, 
                                        retain_graph=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradient_penalty

def readMyImage(cSubject,workingFrom='PC'):
    low_res = []
    high_res = []
    
    
    
    restingState='RestingState-7T'
    
    rFMRIrestList = ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']
    
    for rest in rFMRIrestList:
        if workingFrom == 'PC':
            if restingState=='RestingState-7T':
                currFilePathOne = 'E:/fMRI-Dataset/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseOne_gdc_dc.nii.gz'
                currFilePathTwo = 'E:/fMRI-Dataset/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseTwo_gdc_dc.nii.gz'
            else:
                currFilePathOne = 'E:/fMRI-Dataset/'+restingState+'/'+cSubject+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseOne_gdc_dc.nii.gz'
                currFilePathTwo = 'E:/fMRI-Dataset/'+restingState+'/'+cSubject+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseTwo_gdc_dc.nii.gz'
            
        else:
            if restingState=='RestingState-7T':
                currFilePathOne = 'HCP/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseOne_gdc_dc.nii.gz'
                currFilePathTwo = 'HCP/'+restingState+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/'+rest+'_PhaseTwo_gdc_dc.nii.gz'
            else:
                currFilePathOne = 'HCP/'+restingState+'/'+cSubject+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseOne_gdc_dc.nii.gz'
                currFilePathTwo = 'HCP/'+restingState+'/'+cSubject+'/R_fMRI_Preprocessed/'+cSubject+'/MNINonLinear/Results/'+rest+'/PhaseTwo_gdc_dc.nii.gz'
             
                
       
        img = nib.load(currFilePathOne).get_fdata()
        
        
        #normalising input data rfMRI
        min = img.min(axis=(0, 1, 2), keepdims=True)
        max = img.max(axis=(0, 1, 2), keepdims=True)
        img = (img - min) / (max - min)
        img = torch.tensor(img, dtype=torch.float32)
        
        
        print('Size of img is ',img.shape)
       
        kk = torch.unsqueeze(torch.moveaxis(img, 3, 0), dim=0)
        print('Size of low_res  is ',kk.shape)
        low_res.append(torch.unsqueeze(torch.moveaxis(img, 3, 0), dim=0))
        
        
        
       # normalising reference image/MRI
        img = nib.load(currFilePathTwo).get_fdata()
        min = img.min(axis=(0, 1, 2), keepdims=True)
        max = img.max(axis=(0, 1, 2), keepdims=True)
        img = (img - min) / (max - min)
        img = torch.tensor(img, dtype=torch.float32)        
        
        low_res.append(torch.unsqueeze(torch.moveaxis(img, 3, 0), dim=0))
        
    #high res
    if workingFrom == 'PC':
         currFilePathT1 = 'E:/fMRI-Dataset/Structural-7T/'+cSubject+'/MNINonLinear/T1w_restore.1.60.nii.gz'
         currFilePathT2 = 'E:/fMRI-Dataset/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
    else:
         currFilePathT1 = 'HCP/Structural-7T/'+cSubject+'/MNINonLinear/T1w_restore.1.60.nii.gz'
         currFilePathT2 = 'HCP/Structural-7T/'+cSubject+'/MNINonLinear/T2w_restore.1.60.nii.gz'
         
    
    
    
    
  
    img = nib.load(currFilePathT1).get_fdata()
    
    
    
    min = img.min(axis=(0, 1, 2), keepdims=True)
    max = img.max(axis=(0, 1, 2), keepdims=True)
    img = (img - min) / (max - min)
    img = torch.tensor(img, dtype=torch.float32)
    
    print(img.shape)
    
    high_res.append(img.unsqueeze(dim=0).unsqueeze(dim=0))
    
    
    
    img = nib.load(currFilePathT2).get_fdata()
    
    
   
    min = img.min(axis=(0, 1, 2), keepdims=True)
    max = img.max(axis=(0, 1, 2), keepdims=True)
    img = (img - min) / (max - min)
    img = torch.tensor(img, dtype=torch.float32)
    
    print(img.shape)
    
    high_res.append(img.unsqueeze(dim=0).unsqueeze(dim=0))
    
    
    
        
        
        
    return torch.vstack(low_res), high_res
    
    
    


def read_img(subject, data_path):
    low_res = []
    high_res = []

    for file in os.listdir(Path(data_path, subject)):
        target = True if 'restore' in file else False
        img = nib.load(Path(data_path, subject, file)).get_fdata()
        min = img.min(axis=(0, 1, 2), keepdims=True)
        max = img.max(axis=(0, 1, 2), keepdims=True)
        img = (img - min) / (max - min)
        img = torch.tensor(img, dtype=torch.float32)

        if target:
            high_res.append(img.unsqueeze(dim=0).unsqueeze(dim=0))

        else:
            low_res.append(torch.unsqueeze(torch.moveaxis(img, 3, 0), dim=0))

    return torch.vstack(low_res), high_res


def getMy_subject_loader(subjects, data_path='fMRI-Dataset/'):
    for subject in subjects:        
        images , (T1,_) = readMyImage(subject)
        print('Data type of images ',type(images))
        yield images[:6, :, :, :, :], images[6:, :, :, :, :], T1, subject

def get_subject_loader(subjects, data_path='fMRI-Dataset/'):
    for subject in subjects:
        images, (T1, _) = read_img(subject, data_path)
        yield images[:6, :, :, :, :], images[6:, :, :, :, :], T1, subject


def visualise(images, name, path='.', slice=(42, 82)):
    if images.size(0) > 1:
        for i, img in enumerate(images):
            ax = plt.subplot()
            img = img[:, slice[0]:slice[1], :, :]
            x_grid = make_grid(img.permute(1, 0, 2, 3))
            x_grid = x_grid.permute(1, 2, 0).numpy()

            ax.imshow(x_grid)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f'{name} {i + 1}', fontsize=5)

            plt.savefig(Path(path, f'{name} {i + 1}.jpg'), dpi=1000, bbox_inches='tight')
            plt.close()

    else:
        ax = plt.subplot()
        images = images.squeeze(dim=0)[:, slice[0]:slice[1], :, :]
        x_grid = make_grid(images.permute(1, 0, 2, 3))
        x_grid = x_grid.permute(1, 2, 0).numpy()

        ax.imshow(x_grid)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=5)
        plt.savefig(Path(path, f'{name}.jpg'), dpi=1000, bbox_inches='tight')
        plt.close()

