import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from datetime import datetime

import os

sys.path.insert(1, './')


from util.metrics import *
from projects.DCGANModified.modelSigmoid import Generator, Discriminator

from util.HCPDatasetSubjectTrainValTest import *
from util.memoryManagement import *
from util.GetF_MRIDataAdvance import *
workingFrom = 'UNI'
epochs = 35000#18000#
batch_size=16
visualise_step = 500
lr=0.0002

RUN_TYPE='SUBJECT_SPECIFIC'
#RUN_TYPE='GENERALIZE'# comment line 35 and uncomment line 36 for Generalised training
 
    
def main_fun(rank, world_size):
    setup(rank, world_size)
    # Global variables
    device = rank
    
    
    best_val_ssimMRI = -np.inf
    #logs = []
    
    '''
    # comment line 58 and uncomment this block for Generalised training
    subjects= ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
                  '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
                '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
                  '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
                   '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542',
                   '177140','177645','178142','178243','178647','180533','181232','182436','182739','186949',
                   '187345','191033','191336','191841','192641','200311','201515','203418','209228','214019',
                   '214524','221319','239136','246133','249947','257845','263436', '283543', '318637', '320826',
                   '330324', '346137', '352738', '360030', '380036', '381038', '385046', '389357', '393247', '395756']
    '''

    subjects=['100610']# data for subject-specific training

    

         
        
        
    
    
    rFMRIrestList=['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']
    
    # Training dataset
    data, ansT1, ansT2 = load_HCPDataset_3_1CH(workingFrom,folders=subjects,rFMRIrestList=rFMRIrestList,normalize=True)
    
   
        
    
    #print('size of trainT1 : ', train_img.shape)
    myDataset = HCPDatasetSubjectTrainValTest(data, ansT1, None)
    
    mySampler = DistributedSampler(myDataset,num_replicas=world_size, rank=rank)
   
    myLoader = DataLoader(dataset=myDataset, batch_size=batch_size, sampler=mySampler,shuffle=False)
    
    
   
    
    g = Generator().to(device)
    d = Discriminator().to(device)
    optim_g = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d = optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    one = torch.tensor(1, dtype=torch.float32).to(device)
    mone = one * -1
    
    
    
    #print ('loader is define')
    if rank == 0:
        # Define the writer
        nowTime = datetime.now()
        os.system("fuser 6006/tcp -k")
        if RUN_TYPE=='SUBJECT_SPECIFIC':
            logdir = os.path.join("newLogsDCGAN/DCGAN_SubSpc_allMetrics_TrainTestValFinal_35KEpoch", nowTime.strftime("%Y%m%d-%H%M%S"))

        else:
            logdir = os.path.join("newLogsDCGAN/DCGAN_Generalise_allMetrics_TrainTestVal_7KEpoch", nowTime.strftime("%Y%m%d-%H%M%S"))

        writer = SummaryWriter(logdir)
        
        os.mkdir(logdir+'/CheckPoint')
        os.mkdir(logdir+'/Images')
        os.mkdir(logdir+'/logs')
    
    
    
    
    
    
    
    
    
    
    # WGAN with gradient penalty implementation
    for epoch in range(epochs + 1):
        train_mae = 0.0
        train_mse = 0.0
        train_nmse = 0.0
        train_psnr = 0.0
        train_ssimMRI = 0.0
        train_diceCoef = 0.0
        
        val_mae = 0.0
        val_mse = 0.0
        val_nmse = 0.0
        val_psnr = 0.0
        val_ssimMRI = 0.0
        val_diceCoef =0.0
        
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
        
        # Subject training
        #for train_images, val_images, T1, subject in subject_loader:
        for subject_idx,( train_images, val_images,test_images, T1) in enumerate(myLoader):           
            train_images = train_images.to(rank)
            val_images = val_images.to(rank)
            T1 = T1.to(rank)
            g.train()
            for img in train_images:
               
                T1 = T1.to(device)
    
                # Discriminator training
                for p in d.parameters():
                    p.requires_grad = True
    
                for _ in range(5):
                    d.zero_grad()
    
                    T1 = T1.requires_grad_(True)
                    real_out = d(T1)
                    real_out = real_out.mean()
                    real_out.backward(mone)
    
                    img = img.requires_grad_(False)
                    T1_fake = g(img)
                    T1_fake = T1_fake.data.requires_grad_(True)
    
                    fake_out = d(T1_fake)
                    fake_out = fake_out.mean()
                    fake_out.backward(one)
    
                    gradient_penalty = get_gradient_penalty(T1.data, T1_fake.data, d,rank)
                   
                    gradient_penalty.backward(one)
    
                    optim_d.step()
    
                # Generator training
                for p in d.parameters():
                    p.requires_grad = False
    
                g.zero_grad()
    
                img = img.requires_grad_(True)
                T1_fake = g(img)
                fake_out = d(T1_fake)
    
                fake_out = fake_out.mean()
                fake_out.backward(mone)
    
                optim_g.step()
                
    
                
                
                
                train_mae += mae(T1, T1_fake.data).item()
                train_mse += mse(T1, T1_fake.data).item()
                train_nmse += nmse(T1, T1_fake.data).item()
                train_psnr += psnr(T1, T1_fake.data).item()
                train_ssimMRI += ssim3D(T1, T1_fake.data).item()
                
                for numTrain in range(train_images.size(0)):
                    train_diceCoef += dice_coefficient(T1.cpu().detach().numpy(), T1_fake.data[numTrain,:,:,:,:].cpu().detach().numpy())
                
                
                
                
                
                
                
                
                
    
            # Subject validation 
            dist.barrier()
            g.eval()
            for i, img in enumerate(val_images):
            
                with torch.no_grad():
                    T1_fake = g(img)
                    
                    val_mae += mae(T1, T1_fake.data).item()
                    val_mse += mse(T1, T1_fake.data).item()
                    val_nmse += nmse(T1, T1_fake.data).item()
                    val_psnr += psnr(T1, T1_fake.data).item()
                    val_ssimMRI += ssim3D(T1, T1_fake.data).item()
                    
                    for numVal in range(val_images.size(0)):
                        val_diceCoef += dice_coefficient(T1.cpu().detach().numpy(), T1_fake.data[numVal,:,:,:,:].cpu().detach().numpy())
    
    
                    
                    # Visualise fake images every visualise_step
                    if epoch % visualise_step == 0:
                        visualise(T1_fake.cpu().data, f'{epoch}_{subject_idx}_T1_fake_{i}',path=logdir+'/Images')
                        visualise(T1.cpu().data, f'{epoch}_{subject_idx}_Real_{i}',path =logdir+'/Images')
                    
    
        # Epoch performance    
        train_mae /= (train_images.size(0) * len(subjects))
        train_mse /= (train_images.size(0) * len(subjects))
        train_nmse /= (train_images.size(0) * len(subjects))
        train_psnr /= (train_images.size(0) * len(subjects))
        train_ssimMRI /= (train_images.size(0) * len(subjects))
        
        train_diceCoef /=(train_images.size(0)* len(subjects))
        
        writer.add_scalar("training mae", train_mae , global_step=epoch)                        
        writer.add_scalar("training mse", train_mse , global_step=epoch)
        writer.add_scalar("training nmse", train_nmse , global_step=epoch)
        writer.add_scalar("training psnr", train_psnr , global_step=epoch)
        writer.add_scalar("training ssimMRI", train_ssimMRI, global_step=epoch)
        
        writer.add_scalar("training diceCoeff ", train_diceCoef, global_step=epoch)
        
        
        
        
        
        val_mae /= (val_images.size(0)* len(subjects))
        val_mse /= (val_images.size(0)* len(subjects))
        val_nmse /= (val_images.size(0)* len(subjects))
        val_psnr /= (val_images.size(0)* len(subjects))        
        val_ssimMRI /= (val_images.size(0)* len(subjects))
        
        val_diceCoef /=(val_images.size(0)* len(subjects))
        
        # Write validation performance
        writer.add_scalar("validation mae", val_mae , global_step=epoch)                        
        writer.add_scalar("validation mse", val_mse , global_step=epoch)
        writer.add_scalar("validation nmse", val_nmse , global_step=epoch)
        writer.add_scalar("validation psnr", val_psnr , global_step=epoch)
        writer.add_scalar("validation ssimMRI", val_ssimMRI, global_step=epoch)
        
        writer.add_scalar("validation diceCoeff ", val_diceCoef, global_step=epoch)
        
        
        
        
        #logs.append((train_ssim, val_ssim))
        
        
        
        
        
            
    
        # Save the generator if validation performance is improving
        if val_ssimMRI > best_val_ssimMRI:
            best_val_ssimMRI = val_ssimMRI
            best_epoch = epoch
            
            torch.save(g.state_dict(), logdir+'/CheckPoint/checkpoint.pt')
            
            
       
        print(f"Rank {rank}, Epoch [{epoch} / {epochs}], "\
                  f"Train SSIM : {train_ssimMRI:.4f},"\
                  f"MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, PSNR:{train_psnr:.4f}," 
                  f"Best validation ssimMRI: {best_val_ssimMRI:.4f}, DiceCoefficient:{train_diceCoef:.4f}")
    writer.flush()
    writer.close()
    
        # Save logs
        #torch.save(logs, logdir+'/Logs/logs.pt')
    
    dist.barrier()
    cleanup(rank)


    
def cleanup(rank):
    dist.destroy_process_group()
    print(f"Rank {rank} is done.")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '2222'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
#main_fun('cpu',1)


if __name__ == "__main__":
    torch.manual_seed(1234)
    world_size =  torch.cuda.device_count()
    world_size=1
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        print('Running rank ',rank)
        p = mp.Process(target=main_fun, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
      
        
        
        
        


