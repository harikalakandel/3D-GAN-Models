# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:33:39 2023

@author: Harikala
"""

import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from sklearn import preprocessing
import matplotlib.pyplot as plt


from collections import defaultdict, namedtuple
from typing import List


import numpy as np

import os
import scipy.misc

import imageio

import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()


TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "cnt"])



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf1.placeholder(tf1.string)
    im_tf = tf1.image.decode_image(image_str)

    sess = tf1.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf1.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    print(v.tag)
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    plt.imshow(im)
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    imageio.imwrite(output_fn, im)
                    count += 1 



#save_images_from_event("H:/testInGPU_Cluster/chkMe/20231201-222700/events.out.tfevents.1701469620.3038995fd37b.95.0",'Real_complete','H:/testInGPU_Cluster/Outputs/SaveImages')

EXPERIMENT_NAME='SGAN_Gen_TrainTestValMetricBatch1_35HunEpoch'
DISPLAY_EXPERIMENT_NAME ='CycleGAN'#'DCGAN_Sigmoid'#'CycleGAN'
#DISPLAY_EXPERIMENT_NAME = EXPERIMENT_NAME
LOG_TIME='20240213-203629'
EVENT_NAME = 'events.out.tfevents.1707856589.94d667a4809a.158.0'

caseType=['training ','validation ']
APPLY_NORMAILZATION = True


gen_loss=[]
dis_loss=[]




allMetricsName=['MSE','MAE','NMSE','PSNR','SSIM','Dice Coefficient']


SAVE_LOCATION='H:/testInGPU_Cluster/Evaluation'
#eventFile = 'H:/testInGPU_Cluster/newLogs/'+EXPERIMENT_NAME+'/'+LOG_TIME+'/'+EVENT_NAME
eventFile = 'H:/testInGPU_Cluster/newLogsGen/'+EXPERIMENT_NAME+'/'+LOG_TIME+'/'+EVENT_NAME

#H:\testInGPU_Cluster\newLogsGen\SGAN_Gen_TrainTestValMetricBatch1_35HunEpoch\20240213-203629
#H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_TrainTestValBatch16Epoch25K\20240214-133746
#H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_allMetrics_TrainTestVal_35KEpoch\20240217-183433
#check if folder for plots exist, and if not exit create a new folder
if not (os.path.isdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME)):
    os.mkdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME)

if not (os.path.isdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME+'/Plots')):
    os.mkdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME+'/Plots')
        
SAVE_LOCATION = SAVE_LOCATION+'/'+EXPERIMENT_NAME+'/Plots'

#for e in tf.compat.v1.train.summary_iterator("H:/testInGPU_Cluster/newLogs/Generalise_withEval_48WholeImage_5KEpoch_SGD/20240114-104158/events.out.tfevents.1705228918.13cb5276be10.158.0"):

try:
    for e in tf.compat.v1.train.summary_iterator(eventFile):
        for v in e.summary.value:
                if v.tag.strip()=='gen_loss':
                    gen_loss.append(v.simple_value)
                if v.tag.strip()=='dis_loss':
                    dis_loss.append(v.simple_value)
                    
    # normalised to plot result between 0 and 1
    if APPLY_NORMAILZATION:
        gen_loss= NormalizeData(gen_loss)
        dis_loss= NormalizeData(dis_loss)   
            
    
    plt.plot(gen_loss,label='Generator Loss')
    plt.plot(dis_loss,label='Discriminator Loss')
    plt.title( DISPLAY_EXPERIMENT_NAME + ' DISC/GEN Loss')
    plt.legend()
    plt.savefig(SAVE_LOCATION+'/GenDiscLoss.jpg')
    plt.show()
except:
    print('Generator / Discriminator Loss not saved....')

#allMetricsName=['mse','mae','nmse','psnr','ssimMRI']
allMetricsName=['mse','mae','psnr','ssimMRI','diceCoeff']
allMetrics=[[[],[]] for j in range(len(allMetricsName))]

                
for i in range(len(allMetricsName)):    
    for j in range(len(caseType)):
        tmpData =[]
        for e in tf.compat.v1.train.summary_iterator(eventFile):
            for v in e.summary.value:           
                if v.tag.strip()==caseType[j]+allMetricsName[i]:
                    tmpData.append(v.simple_value)
        allMetrics[i][j]=tmpData
             
                    
         
            
               # print(v.tag)
               #print(v.simple_value)

        
#########################################################################################



  
    
    
for i in range(len(allMetrics)):
    #plot different metrics individually
    plt.plot(allMetrics[i][0],label='Train '+allMetricsName[i])
    plt.plot(allMetrics[i][1],label='Validate '+allMetricsName[i])
    
    #difference
    #dff = [allMetrics[i][0][j]-allMetrics[i][1][j] for j in range(len(allMetrics[i][0]))]
    #plt.plot(dff,label='difference')
    plt.title(DISPLAY_EXPERIMENT_NAME+' '+ allMetricsName[i])
   
    plt.legend()
    plt.savefig(SAVE_LOCATION+'/'+allMetricsName[i]+'.jpg')
    plt.show()
    
if APPLY_NORMAILZATION:
    for i in range(len(allMetricsName)):
        for j in range(len(caseType)):
            allMetrics[i][j] =NormalizeData(allMetrics[i][j])
    
#get plot for train and validation metrics
for j in range(2):
    for i in range(len(allMetricsName)):
        #print('Value of i', i, ' Value of j ',j,'Number of data ',len())
        plt.plot(allMetrics[i][j],label=allMetricsName[i])
    
    plt.title(DISPLAY_EXPERIMENT_NAME+' '+caseType[j])
    plt.legend()
    plt.savefig(SAVE_LOCATION+'/'+caseType[j]+'Plot.jpg')
    
    plt.show()
    
   