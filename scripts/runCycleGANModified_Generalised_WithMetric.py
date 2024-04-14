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

#sys.path.insert(1, 'H:/testInGPU_Cluster')
from util.metrics import *
from projects.CycleGan.utils import  get_gradient_penalty
from projects.CycleGan.utils import ssim3D, visualise
from projects.CycleGan.models import Generator, Discriminator

from util.HCPDatasetSubjectV1 import *
from util.memoryManagement import *
from util.GetF_MRIDataAdvance import *
workingFrom = 'UNI'
batch_size=1#16
epochs = 5000#3500
visualise_step = 200
lr=0.0002

RUN_TYPE='SUBJECT_SPECIFIC'
#RUN_TYPE='GENERALIZE' # comment line 38 and uncomment line 39 for Generalised training
    
def main_fun(rank, world_size):
    setup(rank, world_size)
    # Global variables
    device = rank
    
    
    best_val_ssim = -np.inf
    logs = []
    

    subjects=['100610']
    
    '''
    # comment line 51 and uncomment this block for Generalised training
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
    
    rfMRIrestList=['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']
    
    # Training dataset
    data, ansT1w, ansT2w = load_HCPDataset_3_1CH(workingFrom,folders=subjects,rFMRIrestList=rfMRIrestList,normalize=True)
    
   
    myDataset = HCPDatasetSubjectV1(data, ansT1w, None)
   
    mySampler = DistributedSampler(myDataset,num_replicas=world_size, rank=rank)
    
    myLoader = DataLoader(dataset=myDataset, batch_size=1, sampler=mySampler,shuffle=False)
    
    
   
    
    g = Generator().to(device)
    d = Discriminator().to(device)
    optim_g = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d = optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    one = torch.tensor(1, dtype=torch.float32).to(device)
    mone = one * -1
    
    
    if rank == 0:
        # Define the writer
        nowTime = datetime.now()
        os.system("fuser 6006/tcp -k")
        if RUN_TYPE=='SUBJECT_SPECIFIC':
            logdir = os.path.join("newLogsCycleGAN/SGAN_SubSpc_TrainTestValBatch1", nowTime.strftime("%Y%m%d-%H%M%S"))

        else:
            logdir = os.path.join("newLogsCycleGAN/CycleGAN_Genneralise_TrainTestValMetricBatch1_5KEpoch", nowTime.strftime("%Y%m%d-%H%M%S"))

        writer = SummaryWriter(logdir)
    
    
    
        os.mkdir(logdir+'/GenCheckPoint')
        os.mkdir(logdir+'/Images')
        os.mkdir(logdir+'/logs')
    
    
    
    
    
    
    
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
        # Create subject loader
        
    
        # Subject training
        #for train_images, val_images, T1w, subject in subject_loader:
        for subject_idx,( train_images, val_images, T1w) in enumerate(myLoader):
            
            train_images = train_images.to(rank)
            val_images = val_images.to(rank)
            T1w = T1w.to(rank)
            g.train()
            for img in train_images:
                T1w = T1w.to(device)
    
                # Discriminator training
                for p in d.parameters():
                    p.requires_grad = True
    
                for _ in range(5):
                    d.zero_grad()
    
                    T1w = T1w.requires_grad_(True)
                    real_out = d(T1w)
                    real_out = real_out.mean()
                    real_out.backward(mone)
    
                    img = img.requires_grad_(False)
                    T1w_fake = g(img)
                    T1w_fake = T1w_fake.data.requires_grad_(True)
    
                    fake_out = d(T1w_fake)
                    fake_out = fake_out.mean()
                    fake_out.backward(one)
                    
                    
                    # gradient penalty is implementation    
                    gradient_penalty = get_gradient_penalty(T1w.data, T1w_fake.data, d,rank)
                    gradient_penalty.backward(one)
    
                    optim_d.step()
    
                # Generator training
                for p in d.parameters():
                    p.requires_grad = False
    
                g.zero_grad()
    
                img = img.requires_grad_(True)
                T1w_fake = g(img)
                fake_out = d(T1w_fake)
    
                fake_out = fake_out.mean()
                fake_out.backward(mone)
    
                optim_g.step()
    
               
                train_mae += mae(T1w, T1w_fake.data).item()
                train_mse += mse(T1w, T1w_fake.data).item()
                train_nmse += nmse(T1w, T1w_fake.data).item()
                train_psnr += psnr(T1w, T1w_fake.data).item()
                train_ssimMRI += ssim3D(T1w, T1w_fake.data).item()
                for numTrain in range(train_images.size(0)):
                    train_diceCoef += dice_coefficient(T1w.cpu().detach().numpy(), T1w_fake.data[numTrain,:,:,:,:].cpu().detach().numpy())
                
    
            # Subject validation 
            dist.barrier()
            g.eval()
            for i, img in enumerate(val_images):
               
    
                with torch.no_grad():
                    T1w_fake = g(img)                  
                    val_mae += mae(T1w, T1w_fake.data).item()
                    val_mse += mse(T1w, T1w_fake.data).item()
                    val_nmse += nmse(T1w, T1w_fake.data).item()
                    val_psnr += psnr(T1w, T1w_fake.data).item()
                    val_ssimMRI += ssim3D(T1w, T1w_fake.data).item()
                    for numVal in range(val_images.size(0)):
                        val_diceCoef += dice_coefficient(T1w.cpu().detach().numpy(), T1w_fake.data[numVal,:,:,:,:].cpu().detach().numpy())
    
                    
                    # Visualise fake images every visualise_step
                    if epoch % visualise_step == 0:
                        visualise(T1w_fake.cpu().data, f'{epoch}_{subject_idx}_T1w_fake_{i}',path= logdir+'/Images')
                        visualise(T1w.cpu().data, f'{epoch}_{subject_idx}_Real_{i}',path= logdir+'/Images')
                    
    
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
        writer.add_scalar("validation nmse", val_nmse ,global_step= epoch)
        writer.add_scalar("validation psnr", val_psnr , global_step=epoch)
        writer.add_scalar("validation ssimMRI", val_ssimMRI, global_step=epoch)
        writer.add_scalar("validation diceCoeff ", val_diceCoef, global_step=epoch)
        
        logs.append((train_ssimMRI, val_ssimMRI))
    
        # Save the generator if validation performance is improving
        if val_ssimMRI > best_val_ssim:
            best_val_ssim = val_ssimMRI
            torch.save(g.state_dict(), logdir+'/GenCheckPoint/checkpoint.pt')
            
            
        if rank==0:
           
            print(f'epoch: {epoch}, train ssim: {train_ssimMRI}, val ssim: {val_ssimMRI}, best val ssim: {best_val_ssim},'\
                  f'train dice coefficient:{train_diceCoef}, val dice coefficient:{val_diceCoef} ')
            
    
    # Save logs
    torch.save({'logs': logs}, 'logs.pt')
    
    writer.flush()
    writer.close()
    
    
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
      
        
        
        
        
