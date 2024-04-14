import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, out_channels=64, kernel_size=4, stride=2, padding=1):
        super().__init__()
        
        

        self.downscale = nn.Sequential(
            nn.Conv3d(3, out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.ReLU(),
            nn.Conv3d(out_channels * 2, out_channels * 4, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.ReLU(),
            nn.Conv3d(out_channels * 4, out_channels * 8, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.ReLU(),
            
        )
        self.upscale = nn.Sequential(
           
            nn.ConvTranspose3d(out_channels * 8, out_channels * 4, kernel_size=(kernel_size,kernel_size+1,kernel_size), stride=stride,padding=padding),
            nn.ReLU(),
            nn.ConvTranspose3d(out_channels * 4, out_channels * 2, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.ReLU(),
            nn.ConvTranspose3d(out_channels * 2, out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.ReLU(),
            nn.ConvTranspose3d(out_channels, 1, kernel_size=(kernel_size+1,kernel_size,kernel_size+1), stride=stride,padding=padding),
            #nn.Sigmoid(),
            nn.Tanh(),
           
        )
          
    def forward(self, img):
        
        img = self.downscale(img) 
        img = self.upscale(img)
       
        return img

       
    

class Discriminator(nn.Module):
    def __init__(self, out_channels=64, kernel_size=4, stride=2,  padding=1):
        super().__init__()
        
        self.dis = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding),           
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels * 2, out_channels * 4, kernel_size=kernel_size, stride=stride, padding=padding),          
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels * 4, out_channels * 8, kernel_size=kernel_size, stride=stride, padding=padding),            
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels * 8, 1, kernel_size=kernel_size, stride=stride, padding=0),
        )
        
    def forward(self, img):
        out = self.dis(img)
        
        return out
        