import torch
import sys
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


workingFrom = 'UNI'#'PC'
MODEL_NAME = 'CycleGAN'#'DCGAN_Sigmoid'#'CycleGAN'#'DCGAN_Sigmoid'#'CycleGAN'#'DCGAN_Sigmoid'#'DCGAN' #'CycleGAN' #'DCGAN_TANH' #'SRGAN_New' #'SRGAN_NewB2'
EXPERIMENT_NAME='CycleGAN_Genneralise_TrainTestValMetricBatch16_7KEpoch'
LOG_TIME='20240223-153928'

IS_SUBJECT_SPECIFIC = False #True

if workingFrom == 'PC':
    sys.path.insert(1,'H:/testInGPU_Cluster')    
    SAVE_LOCATION = 'H:/testInGPU_Cluster/Evaluation'
    generator_path = 'H:/testInGPU_Cluster/newLogsSRGAN/'+EXPERIMENT_NAME+'/'+LOG_TIME+'/CheckPoint'
    #generator_path = 'newLogs/'+EXPERIMENT_NAME+'/20240109-181334' # TO CHANGE HERE
   
    
else:
    sys.path.insert(1, './')
    SAVE_LOCATION = 'Evaluation'
    #generator_path = 'newLogsSRGAN/SRGAN_SubSpc_modelNew_25KEpoch/20240304-184715/CheckPoint'
    #generator_path = 'newLogs/'+EXPERIMENT_NAME+'/'+LOG_TIME# +'/CheckPoint' # TO CHANGE HERE
    #generator_path = 'newLogsCycleGAN/CycleGAN_SubSpc_TrainTestValBatch16Epoch25K/20240214-133746/CheckPoint'
    #generator_path = 'newLogsCycleGAN/'+EXPERIMENT_NAME+'/'+LOG_TIME +'/CheckPoint'
    generator_path = 'newLogsCycleGAN/'+EXPERIMENT_NAME+'/'+LOG_TIME +'/GenCheckPoint'
    
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_TrainTestValBatch16Epoch25K\20240214-133746 
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_allMetrics_TrainTestVal_35KEpoch\20240217-183433\CheckPoint
 #H:\testInGPU_Cluster\newLogsSRGAN\SRGAN_SubSpc_modelNew_25KEpoch\20240304-184715
 #H:\testInGPU_Cluster\DockerOut\SGAN_Generalised_WholeImg/Gencheckpoint
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_TrainTestValBatch16Epoch25K\20240214-133746\CheckPoint
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_allMetrics_TrainTestValFinal_50KEpoch\20240227-184550\CheckPoint
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_secondSub-102816_40KEpoch\20240307-152911\CheckPoint
 #H:\testInGPU_Cluster\newLogs\DCGANBased_Generalise_65Subjects_WholeImage_3200EpochFinal\20240207-105213
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_secondSub-102311_40KEpoch\20240307-151619\CheckPoint
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_fourthSub-104416_40KEpoch\20240307-154722\CheckPoint
 
 
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_Genneralise_TrainTestValMetricBatch16_7KEpoch\20240223-153928\GenCheckPoint
 
 
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_102311_Batch16Epoch40K\20240312-074031\CheckPoint
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_102816_Batch16Epoch40K\20240313-114820\CheckPoint
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_104416_Batch16Epoch40K\20240313-115752\CheckPoint
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_105923_Batch16Epoch40K\20240316-161440\CheckPoint
 
 #H:\testInGPU_Cluster\newLogs\Generalise_withEval_48WholeImage_20K_Epoch\20240106-185504
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_fifthSub_105923_40KEpoch\20240307-161131\CheckPoint
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGANBased_GeneraliseFewSubs_50KEpoch\20240302-143030\CheckPoint
 #H:\testInGPU_Cluster\newLogs\Generalise_48WholeImage_15HunEpoch_Adamlr0002\20240131-000954
 #H:\testInGPU_Cluster\newLogs\SubSpc_wholeImge_20KEpoch_Eval_Adam_newModel\20240129-174314
 #H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_allMetrics_TrainTestVal_35KEpoch\20240217-183433  
 #H:\testInGPU_Cluster\newLogs\DCGANBased_Generalise_65Subjects_WholeImage_3200EpochFinal\20240207-105213
 # H:\testInGPU_Cluster\newLogsDCGAN\DCGAN_SubSpc_allMetrics_TrainTestValFinal_18KEpoch\20240218-201015
 #H:\testInGPU_Cluster\newLogsGen\SGAN_Gen_TrainTestValMetricBatch1_35HunEpoch\20240213-203629
 #H:\testInGPU_Cluster\newLogsCycleGAN\CycleGAN_SubSpc_TrainTestValBatch16Epoch25K\20240214-133746
 #H:\testInGPU_Cluster\newLogsGen\SGAN_Gen_TrainTestValMetricBatch1_35HunEpoch\20240213-203629\GenCheckPoint
    
if not (os.path.isdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME)):
    os.mkdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME)

if not (os.path.isdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME+'/Images')):
    os.mkdir(SAVE_LOCATION+'/'+EXPERIMENT_NAME+'/Images')
    
    
SAVE_LOCATION = SAVE_LOCATION+'/'+EXPERIMENT_NAME+'/Images/'
    

#generator_path = 'newLogs/'+EXPERIMENT_NAME+'/20240109-181334' # TO CHANGE HERE
#generator_path = 'newLogs/'+EXPERIMENT_NAME+'/20240131-143404' # TO CHANGE HERE
#generator_path='H:/testInGPU_Cluster'
#generator_path = 'H:/testInGPU_Cluster/DockerOut/SGAN_Generalised_WholeImg'
#generator_path = DockerOut/SGAN_SubSpc_TrainValTest'



#generator_path = 'DockerOut/SGAN_Generalised_WholeImg'

#H:\testInGPU_Cluster\DockerOut\SGAN_SubSpc_TrainValTest


#H:\testInGPU_Cluster\DockerOut\SGAN_SubSpc_TrainValTest
#workingFrom = 'UNI' # TO DO
 # TO DO


if MODEL_NAME == 'CycleGAN':
    from projects.CycleGan.models import Generator, Discriminator
    #from util.HCPDatasetSubjectV1 import *
elif MODEL_NAME == 'DCGAN_Sigmoid':
    from projects.DCGANModified.modelSigmoid import Generator, Discriminator
elif MODEL_NAME == 'SRGAN_New':
    from projects.SRGANModified.modelNew import *
else:
    from projects.SRGANModified.modelNewB2 import *


from util.metrics import *
#from projects.CycleGan.utils import  get_gradient_penalty
#from projects.CycleGan.utils import ssim3D, visualise
from torch.utils.data import DataLoader
from util.GetF_MRIDataAdvance import *

from datetime import datetime
from util.metrics import *
from util.HCPDataset import *

IS_SLICE = False#True

if IS_SLICE==True:
    from util.HCPDatasetT1T2seq import *
    sliceSize_=16
else:
    from util.HCPDataset import *







# Python program to get average of a list 
def Average(lst): 
    return sum(lst) / len(lst) 

def STD(lst):
    mean = sum(lst) / len(lst) 
    variance = sum([((x - mean) ** 2) for x in lst]) / len(lst) 
    res = variance ** 0.5
    return res
    

def load_best_generator(path):
    device = torch.device('cpu')
    params = torch.load(path+'/checkpoint.pt', map_location=device)
    #params = torch.load(path+'/Gencheckpoint.pt', map_location=device)
    try:
        gen_params = {key.replace("module.", ""): value for key, value in params['generator'].items()}
    except:
        #for SGAN
        gen_params = {key.replace("module.", ""): value for key, value in params.items()}
        

    #generator = Generator3To1()
    generator = Generator()
    generator.load_state_dict(gen_params)

    return generator


def get_img(low_res, high_res, nrow, generator):
    [samp, ch, x_range, y_range, z_range]= high_res.shape
    #print('high_res shape ',high_res.shape)
    #print('low_res shape ',low_res.shape)
    img_fake = generator(low_res)
    #print('img_fake shape ',img_fake.shape)
    img_grid_real = make_grid(
        high_res[:, :, int(x_range / 2), :, :], nrow=nrow
    )
    #print('img_grid_real shape ',img_grid_real.shape)
    
    img_grid_fake = make_grid(
        img_fake[:, :, int(x_range / 2), :, :], nrow=nrow
    )
    #print('img_grid_fake shape ',img_grid_fake.shape)

    return img_grid_real, img_grid_fake




'''
All_folders = ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
              '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
               '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
               '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
               '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542',
               '177140','177645','178142','178243','178647','180533','181232','182436','182739','186949',
               '187345','191033','191336','191841','192641','200311','201515','203418','209228','214019',
               '214524','221319','239136','246133','249947','257845','263436']
'''



'''
All_folders = [['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
              '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
               '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
               '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
               '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542'],
               ['177140','177645','178142','178243','178647','180533','181232','182436','182739','186949','187345','191033','191336','191841','192641','200311'],
               ['201515','203418','209228','214019','214524','221319','239136','246133','249947','257845','263436']]
'''
if IS_SUBJECT_SPECIFIC:
    #one extra test Invalid cases, where different subject is provided
    caseName =['Train','Validate','Test','InvalidTest']
    #All_folders = [['100610'],['100610'],['100610'],['102311']] 
    All_folders = [['105923'],['105923'],['105923'],['100610']] 
    #All_folders = [['102816'],['102816'],['102816'],['105923']]
    #All_folders = [['104416'],['104416'],['104416'],['102816']]
    #All_folders = [['102311'],['102311'],['102311'],['104416']]
    All_rFMRIrestList=[['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP'],
                       ['rfMRI_REST3_7T_PA'],
                       ['rfMRI_REST4_7T_AP'],
                       ['rfMRI_REST4_7T_AP']]
else:
    #Only three cases
    caseName =['Train','Validate','Test']
    #three different set of subjects
    
    # else:
    #     allFolders= ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
    #                  '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
    #                  '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
    #                  '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
    #                  '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542',
    #                  '177140','177645','178142','178243','178647','180533','181232','182436','182739','186949',
    #                  '187345','191033','191336','191841','192641','200311','201515','203418','209228','214019',
    #                  '214524','221319','239136','246133','249947','257845','263436', '283543', '318637', '320826',
    #                  '330324', '346137', '352738', '360030', '380036', '381038', '385046', '389357', '393247', '395756']
    #train_folders = ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514']
    
    
    All_folders = [
                        ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
                         '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
                         '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
                         '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
                         '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542'],
                         ['200311','201515','203418','209228','214019'],
                         ['283543', '318637', '320826','330324']
                ]
                   #'177140','177645','178142','178243','178647','180533','181232','182436','182739','186949',
                   #'187345','191033','191336','191841','192641'] 
   
    
    #val_folders = ['200311','201515','203418','209228','214019','214524','221319','239136','246133','249947','257845','263436'] 
    
    #train_folders = ['109123'] # TO DO
    #val_folders = ['109123'] # TO DO
    
    #rFMRIrestList=['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']
    
    
    
    
    #three set of all sessions    
    All_rFMRIrestList=[['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP'],
                       ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP'],
                       ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']]
#nrow = 2 # TO DO

#writer = SummaryWriter(logdir)

  
#All_rFMRIrestList=[['rfMRI_REST2_7T_AP'],['rfMRI_REST2_7T_AP'],['rfMRI_REST2_7T_AP']]#'rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']

'''
All_rFMRIrestList=[['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP'],
                   ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP'],
                   ['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']]
'''
'''
All_rFMRIrestList=[['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP'],
                   ['rfMRI_REST3_7T_PA'],
                   ['rfMRI_REST2_7T_AP']]
'''



#generator = load_best_generator(generator_path)
generator = load_best_generator(generator_path)


for caseId in range(len(caseName)):
    #print('Genertor is :')
    #print(generator)
    #low_res, high_res_T1, _, subjectInfo, sliceInfo = load_T1wT2wSliceV2(workingFrom, sliceSize=sliceSize_, folders=folders)
    
    folders = All_folders[caseId]
    rFMRIrestList = All_rFMRIrestList[caseId]
    
    low_res, high_res_T1, high_res_T2 = load_HCPDataset_3_1CH(workingFrom,folders=folders,rFMRIrestList=rFMRIrestList,normalize=True)
    test_dataset = HCPDataset(low_res, high_res_T1, None)
    
    
    
   
    
    
    all_loader = DataLoader(dataset=test_dataset, batch_size=low_res.shape[0])
    
    ##view all images
    for batch_idx, (low_res, high_res) in enumerate(all_loader):
        #add 1 to make images display more columns than in row
               
        row = int(low_res.shape[0]**0.5)+1
        img_grid_real, img_grid_fake = get_img(low_res, high_res, row, generator)
        plt.imshow(img_grid_real[0,:,:])
        plt.savefig(SAVE_LOCATION+caseName[caseId]+'_real.jpg')
        plt.show()
        plt.imshow(img_grid_fake[0,:,:])
        plt.savefig(SAVE_LOCATION+caseName[caseId]+'_fake.jpg')
        plt.show()

        
    eval_mae=[]
    eval_mse=[]
    eval_nmse=[]
    eval_psnr=[]
    eval_ssimMRI=[]
    eval_diceCoff =[]
    ## evaluate generator
    eval_loader = DataLoader(dataset=test_dataset, batch_size=1)
        
    
       
        
        
        
    for batch_idx, (low_res, high_res) in enumerate(eval_loader):
             
        img_fake = generator(low_res)
      
        eval_mae.append( mae(img_fake, high_res).detach().numpy())
        eval_mse.append( mse(img_fake, high_res).detach().numpy())
        eval_nmse.append( nmse(img_fake, high_res).detach().numpy())
        eval_psnr.append(psnr(img_fake,high_res).detach().numpy())
        eval_ssimMRI.append(ssim3D(img_fake, high_res).detach().numpy())
        eval_diceCoff.append(dice_coefficient(img_fake.detach().numpy(), high_res.detach().numpy()))
       
        
      
        
    
    #plt.show()
    # Printing average of the list 
    #print('mae:')
    print (*eval_mae, sep=",")
    print('mae      :',eval_mae)
    print('mse      :',eval_mse)
    print('nmse     :',eval_nmse)
    print('psnr     :',eval_psnr)
    print('ssimMRI  :',eval_ssimMRI)
    print('Metrices information for case -----------------------------------',caseName[caseId])
    print("Average of the mae                   =", Average(eval_mae)) 
    print("Standard Deviation  of the mae       =", STD(eval_mae)) 
    print("Average of the mse                   =", Average(eval_mse)) 
    print("Standard Deviation  of the mse       =", STD(eval_mse)) 
    print("Average of the nmse                  =", Average(eval_nmse)) 
    print("Standard Deviation  of the nmse      =", STD(eval_nmse)) 
    print("Average of the psnr                  =", Average(eval_psnr)) 
    print("Standard Deviation  of the nmse      =", STD(eval_psnr)) 
    print("Average of the ssim                  =", Average(eval_ssimMRI)) 
    print("Standard Deviation  of the ssim      =", STD(eval_ssimMRI))
    print("Average of the Dice Coefficient      =", Average(eval_diceCoff)) 
    print("Standard Deviation  of the Dice Coefficient =", STD(eval_diceCoff))






    
    

