#model from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/SRGAN/train.py
#training from https://neptune.ai/blog/gan-failure-modes
import torch
import sys
from torch import nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
from datetime import datetime

import os

#print('Current Path is ',os.getcwd())

sys.path.insert(1, './')


from projects.SRGANModified.modelNew import *
from util.metrics import *
from util.HCPDataset import *
from util.memoryManagement import *
from util.GetF_MRIDataAdvance import *



# Global variables
torch.backends.cudnn.benchmark = True
workingFrom = 'UNI'
APPLY_INTERPOLATION = False
AVERAGE_DISC_REAL_FAKE_LOSS=True
SEPERATE_DISC_LOSS_BACKWARD=False
RUN_IN_TPU = False
RUN_TYPE='SUBJECT_SPECIFIC'
batch_size = 1
epochs = 10000
nrow = 1
k = 1
LEARNING_RATE=0.0002
world_size = 1



def main_fun(rank, world_size):
    setup(rank, world_size)
   
   
    #     allFolders= ['100610', '102311', '102816', '104416', '105923','108323', '109123', '111514', '114823','115017',
    #               '115825','116726', '118225', '125525', '126426', '126931', '128935', '130114', '131722', '134627',
    #               '134829','135124','137128','140117','145834','146129','146432', '146735','146937','148133',
    #               '150423','155938','156334','157336','158035','158136','159239','162935','164131','164636',
    #               '165436','167440','169040','169444','169747','171633','172130','173334','175237','176542',
    #               '177140','177645','178142','178243','178647','180533','181232','182436','182739','186949',
    #               '187345','191033','191336','191841','192641','200311','201515','203418','209228','214019',
    #               '214524','221319','239136','246133','249947','257845','263436']


    train_folders = ['102311'] # TO DO---need to add more subjects for generalised training
    val_folders = ['102816'] # TO DO------need to add more subjects for generalised training
    
    rFMRIrestList=['rfMRI_REST1_7T_PA','rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']
    
    # Training dataset
    train_img, train_T1, train_T2 = load_HCPDataset_3_1CH(workingFrom,folders=train_folders,rFMRIrestList=rFMRIrestList,normalize=True)
   
    
    
    train_dataset = HCPDataset(train_img, train_T1, None)
   
    data_sampler = DistributedSampler(train_dataset,num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=data_sampler)
    
   
    #if rank == 0:
    # Define the writer
    nowTime = datetime.now()
    os.system("fuser 6006/tcp -k")
    if RUN_TYPE=='SUBJECT_SPECIFIC':
        logdir = os.path.join("newLogs/SubSpc_wholeImge_10KEpoch_Eval_SGD_newModel", nowTime.strftime("%Y%m%d-%H%M%S"))

    else:
        logdir = os.path.join("newLogs/withEvalWholeImage", nowTime.strftime("%Y%m%d-%H%M%S"))

    writer = SummaryWriter(logdir)
    
    print('writer is define')
    
    
    rFMRIrestList=['rfMRI_REST1_7T_PA']#,'rfMRI_REST2_7T_AP','rfMRI_REST3_7T_PA','rfMRI_REST4_7T_AP']

    # Validation dataset
    val_img, val_T1, val_T2 = load_HCPDataset_3_1CH(workingFrom, folders=val_folders,rFMRIrestList=rFMRIrestList,normalize=True )
    val_dataset = HCPDataset(val_img, val_T1,None)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

    #print(f"Rank{rank} Loader is defined.....")
   
    generator = Generator3To1().to(rank)
   
    discriminator = Discriminator().to(rank)
    
    generator = DDP(generator,device_ids=[rank],output_device=rank)
    discriminator = DDP(discriminator,device_ids=[rank],output_device=rank)
    
    #optim_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    #optim_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    optim_g = optim.SGD(generator.parameters(), lr=LEARNING_RATE, momentum=0.9)
    optim_d = optim.SGD(discriminator.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    

    bce = nn.BCEWithLogitsLoss()
    best_val_ssimMRI = -1.0

    for epoch in range(epochs):
        epoch_loss_gen = 0.0
        epoch_loss_disc = 0.0
        train_mae = 0.0
        train_mse = 0.0
        train_nmse = 0.0
        train_psnr = 0.0
        train_ssimMRI = 0.0
        generator.train()
        
        for batch_idx, (train_low_res, train_imgT1) in enumerate(train_loader):
            
            high_res=train_imgT1
            
            high_res = high_res.to(rank)
            
            b_size = high_res.size(0)
            real_label = torch.full((b_size, 1), 0.95).to(rank)            
            fake_label = torch.full((b_size, 1), 0.05).to(rank)  #my_vector = torch.full((vector_size,), fill_number)
                 
            data_fake = generator(train_low_res)
            
            # Train Discriminator       
               
            optim_d.zero_grad()
           
            output_real = discriminator(high_res)
            
            
            output_fake = discriminator(data_fake.detach())
           
           
            
            
            loss_real = bce(output_real, real_label)
            loss_fake = bce(output_fake, fake_label)          
                                
            if AVERAGE_DISC_REAL_FAKE_LOSS:
                loss_disc = (loss_real + loss_fake) / 2
            else:
                loss_disc = loss_real + loss_fake
                
            loss_disc.backward()
            
            optim_d.step() 
            epoch_loss_disc += loss_disc.item()
                
            #Train Generator...............................................
            optim_g.zero_grad()
            output = discriminator(data_fake)    
            #output=output.cpu()  ##???
            gen_loss = bce(output, real_label)

            gen_loss.backward()
            optim_g.step()
            epoch_loss_gen += gen_loss.item()

            #if rank == 0:
               
            
           
            train_mae += mae(data_fake.detach(), high_res)
            train_mse += mse(data_fake.detach(), high_res)
            train_nmse += nmse(data_fake.detach(), high_res)
            train_psnr += psnr(data_fake.detach(),high_res)
            train_ssimMRI += ssim3D(data_fake.detach(), high_res)
                
        # Evaluate Generator
        dist.barrier()  
        if rank == 0:
            val_mae = 0.0
            val_mse = 0.0
            val_nmse = 0.0
            val_psnr = 0.0
            val_ssimMRI = 0.0
            generator.eval()
            with torch.no_grad():
                for low_res, imgT1 in val_loader:
                    high_res = imgT1

                    high_res = high_res.to(rank)
                    low_res = low_res.to(rank)

                    data_fake = generator(low_res)

                    val_mae += mae(data_fake.detach(), high_res)
                    val_mse += mse(data_fake.detach(), high_res)
                    val_nmse += nmse(data_fake.detach(), high_res)
                    val_psnr += psnr(data_fake.detach(),high_res)
                    val_ssimMRI += ssim3D(data_fake.detach(), high_res)

            # Write validation performance
            writer.add_scalar("validation mae", val_mae / len(val_loader), global_step=epoch)                        
            writer.add_scalar("validation mse", val_mse / len(val_loader), global_step=epoch)
            writer.add_scalar("validation nmse", val_nmse / len(val_loader), global_step=epoch)
            writer.add_scalar("validation psnr", val_psnr / len(val_loader), global_step=epoch)
            writer.add_scalar("validation ssimMRI", val_ssimMRI / len(val_loader), global_step=epoch)

            # Save best generator
            if (val_ssimMRI / len(val_loader)) >= best_val_ssimMRI:
                best_val_ssimMRI = val_ssimMRI / len(val_loader)

                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    }, Path(logdir, 'checkpoint.pt')
                )

            # Write training performance
            writer.add_scalar("gen_loss", epoch_loss_gen, global_step=epoch)
            writer.add_scalar("dis_loss", epoch_loss_disc, global_step=epoch)
            writer.add_scalar("training mae", train_mae / len(train_loader), global_step=epoch)                        
            writer.add_scalar("training mse", train_mse / len(train_loader), global_step=epoch)
            writer.add_scalar("training nmse", train_nmse / len(train_loader), global_step=epoch)
            writer.add_scalar("training psnr", train_psnr / len(train_loader), global_step=epoch)
            writer.add_scalar("training ssimMRI", train_ssimMRI / len(train_loader), global_step=epoch)
            
            
            SAVE_IMAGE_PER_EPOCH = 50
            
            if epoch % SAVE_IMAGE_PER_EPOCH == 0:
                #fake = generator(low_res)
                [samp, ch,x_range,y_range,z_range]=train_imgT1.shape
              
                data_fake = generator(train_low_res)
               
                
                
                img_grid_real = torchvision.utils.make_grid(
                    #real[:32], normalize=True
                    #mri[0,0,:,:,int(z_range/2)],normalize=True
                    train_imgT1[:,:,:,:,int(z_range/2)],nrow=nrow
                )
                
                
                img_grid_fake = torchvision.utils.make_grid(
                    #fake[0,0,:,:,int(z_range/2)], normalize=True
                    data_fake[:,:,:,:,int(z_range/2)],nrow=nrow
                )
            
            
                writer.add_image("Real", img_grid_real, global_step=int(epoch/SAVE_IMAGE_PER_EPOCH))
                writer.add_image("Fake", img_grid_fake, global_step=int(epoch/SAVE_IMAGE_PER_EPOCH))
            
            
            
            
            
            
            
            
            
            

            print(f"Rank {rank}, Epoch [{epoch} / {epochs}] Batch {batch_idx} / {len(train_loader)}, "\
                  f"Loss D: {epoch_loss_disc:.4f}, loss G: {epoch_loss_gen:.4f}, Epoch {epoch}, "\
                  f"Best validation ssimMRI {best_val_ssimMRI}")
        
    print(f"Rank{rank} Training Completed...............")

    if rank == 0:
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


if __name__ == "__main__":
    torch.manual_seed(1234)
    world_size =1 # torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        print('Running rank ',rank)
        p = mp.Process(target=main_fun, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

