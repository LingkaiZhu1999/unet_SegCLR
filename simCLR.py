import torch
from loss.nt_xent import NTXentLoss
import torch.nn.functional as F
from loss.BCEDiceLoss import BCEDiceLoss
from unet import Unet_SimCLR, Encoder_SimCLR, Unet
from tqdm import tqdm
import time
from metrics import compute_dice, Dice, AverageLoss
from dataset import BratsTrainContrastDataset, BratsSuperviseTrainDataset, BratsTestDataset
from glob import glob
import albumentations as A
import os
import numpy as np
from utils import draw_training, draw_training_loss, draw_training_joint_on_source
import pickle
import random 
from torch.cuda.amp.grad_scaler import GradScaler
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def __len__(self):
        return len(self.memory)

    def push(self, z1, z2):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (z1, z2)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Sample batch size zs from the replay memory and return them
        '''
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))

class SimCLR(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:1')
        self.nt_xent_loss = NTXentLoss(self.device, self.args.batch_size, self.args.temperature, use_cosine_similarity=True)

    def calculate_contrast_loss(self, model, x1, x2):
        z1, predict1 = model(x1)
        z2, predict2 = model(x2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)        
        contrast_loss = self.nt_xent_loss(z1.cpu(), z2.cpu())

        return contrast_loss

    def CL_train(self):
        train_transform = A.Compose([
        # A.Resize(200, 200),
        # A.CropNonEmptyMaskIfExists(height=150, width=150),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.5),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.ColorJitter(brightness=0.6)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

        train_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain_target}/All/*')
        # val_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain}/Test/*')

        train_dataset = BratsTrainContrastDataset(train_paths, augmentation=train_transform)
        # val_dataset = BratsSuperviseTrainDataset(val_paths, augmentation=None)

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.args.batch_size,shuffle=True,pin_memory=True,drop_last=False)
        # val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,pin_memory=True,drop_last=False)

        model = Encoder_SimCLR(in_channel=self.args.input_channel).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1) # Decrease the learning rate
        loss_contrast_avg_ = []
        compute_metric = AverageLoss()
        trigger = 0
        smallest_loss = 100
        for epoch in range(self.args.epochs):
            for i, data in tqdm(enumerate(train_loader, start=1), total=len(train_loader)):

                image1, image2, _, _ = data

                image1 = image1.to(self.device)
                image2 = image2.to(self.device)

                z1 = model(image1)
                z2 = model(image2)

                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)        
                contrast_loss = self.nt_xent_loss(z1.cpu(), z2.cpu()) 
                # contrast_loss = self.calculate_contrast_loss(model, image1, image2)

                total_loss =  contrast_loss 
                
                # loss_.append(total_loss)
                # metrics 
                # diceValue1 = compute_dice(predict1, label1)
                # diceValue2 = compute_dice(predict2, label2)
                # avg_dice = (diceValue1 + diceValue2) / 2 
                # backprop
                
                optimizer.zero_grad()
                total_loss.backward()

                # update weights

                optimizer.step()

                compute_metric.update(contrast_loss)
            trigger += 1
            loss_contrast_avg = compute_metric.compute()
            loss_contrast_avg = loss_contrast_avg.item()
            loss_contrast_avg_.append(loss_contrast_avg)
            print('Training: Lr: {} Epoch [{:0>3} / {:0>3}] Contrastive Loss {:.4f}'.format(
                optimizer.param_groups[0]['lr'], epoch + 1, self.args.epochs, loss_contrast_avg
            ))
            draw_training_loss(epoch + 1, loss_contrast_avg_)
            compute_metric.reset()
            if epoch > self.args.warm_up:
                scheduler.step()
            if loss_contrast_avg < smallest_loss:
                smallest_loss = loss_contrast_avg
                torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_contrastive_model.pt'))

    def joint_train_on_source_and_target(self):
        # memory = ReplayMemory(500)
        train_transform = A.Compose([
        # A.Resize(200, 200),
        # A.CropNonEmptyMaskIfExists(height=150, width=150),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.5),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.ColorJitter(brightness=0.6)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

        train_source_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain_source}/Train/*')
        train_target_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain_target}/All/*')
        val_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain_target}/All/*')

        train_source_dataset = BratsTrainContrastDataset(train_source_paths, augmentation=train_transform)
        train_target_dataset = BratsTrainContrastDataset(train_target_paths, augmentation=train_transform)
        val_dataset = BratsTrainContrastDataset(val_paths, augmentation=train_transform)

        train_source_loader = torch.utils.data.DataLoader(train_source_dataset,batch_size=self.args.batch_size,shuffle=True,pin_memory=True,drop_last=False)
        train_target_loader = torch.utils.data.DataLoader(train_target_dataset,batch_size=self.args.batch_size,shuffle=True,pin_memory=True,drop_last=False)

        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.args.batch_size,shuffle=False,pin_memory=True,drop_last=False)

        model = Unet_SimCLR(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
        model.apply(weights_init_kaiming) 
        criterion = BCEDiceLoss() # supervise loss function 

        optimizer = torch.optim.Adam(model.parameters(), self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1) # Decrease the learning rate

        start_time = time.time()
        end_time = time.time()
        best_val_loss = 10000
        best_val_dice = 0
        dice_avg_train = []
        val_loss_ = []
        loss_supervise_avg_ = []
        loss_contrast_avg_ = []
        compute_metric = Dice()
        trigger = 0
        model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)
        for epoch in range(self.args.epochs):
            for i, data in tqdm(enumerate(train_target_loader, start=1), total=len(train_target_loader)):

                image1, image2, _, _ = data

                image1 = image1.to(self.device)
                image2 = image2.to(self.device)

                # image11 = image11.to(self.device)
                # image22 = image22.to(self.device)

                z1, _ = model(image1)
                z2, _ = model(image2)
                # z11, _ = model(image11)
                # z22, _ = model(image22)
                image_source1, image_source2, label_source1, label_source2 = next(iter(train_source_loader))
                image_source1 = image_source1.to(self.device)
                image_source2 = image_source2.to(self.device)
                z11, predict_source1 = model(image_source1)
                z22, _ = model(image_source2)


                supervise_loss = criterion(predict_source1.cpu(), label_source1.cpu())
                # supervise_loss2 = criterion(predict2.cpu(), label2.cpu())

                # supervise_loss = (supervise_loss1) / 2

                z1 = torch.concat((z1, z11), dim=0)
                z2 = torch.concat((z2, z22), dim=0)   
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)

                # memory.push(z1=z1, z2=z2)     
                
                
                contrast_loss = self.nt_xent_loss(z1.cpu(), z2.cpu()) / 2
                # contrast_loss = self.calculate_contrast_loss(model, image1, image2)

                total_loss =  self.args.lam * supervise_loss + contrast_loss
                
                # loss_.append(total_loss)
                # metrics 
                # diceValue1 = compute_dice(predict1, label1)
                # diceValue2 = compute_dice(predict2, label2)
                # avg_dice = (diceValue1 + diceValue2) / 2 
                # backprop
                
                optimizer.zero_grad()
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # total_loss.backward()

                # update weights

                optimizer.step()

                compute_metric.update(predict=predict_source1, label=label_source1, loss_sup=supervise_loss, loss_con=contrast_loss)
                # compute_metric.update(predict=predict2, label=label2, loss_sup=supervise_loss2, loss_con=contrast_loss)
            trigger += 1
            dice_avg, loss_supervise_avg, loss_contrast_avg = compute_metric.compute()
            dice_avg = torch.mean(dice_avg).item()
            loss_supervise_avg = loss_supervise_avg.item()
            loss_contrast_avg = loss_contrast_avg.item()
            dice_avg_train.append(dice_avg)
            loss_supervise_avg_.append(loss_supervise_avg)
            loss_contrast_avg_.append(loss_contrast_avg)
            print('Training: Lr: {} Epoch [{:0>3} / {:0>3}] Total Loss: {:.4f} Supervise Loss: {:.4f} Contrastive Loss {:.4f} Dice: {:.4f}'.format(
                optimizer.param_groups[0]['lr'], epoch + 1, self.args.epochs, loss_supervise_avg + loss_contrast_avg, loss_supervise_avg, loss_contrast_avg, dice_avg
            ))
            
            compute_metric.reset()
            if epoch % self.args.validate_frequency == 0:
                torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', f'{epoch}.pt'))
                start_time = time.time()
                val_loss = self.validate_contrastive(model, val_loader)
                val_loss_.append(val_loss)
                end_time = time.time()
                draw_training(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_train, val_loss_, fig_name=self.args.name)
                if val_loss < best_val_loss:
                    trigger = 0
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_contrast_model.pt'))
                    print("Saving best model")
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s \n".format(
                    epoch + 1, self.args.epochs, val_loss,
                    (end_time - start_time)))
            # # early stopping
            # if self.args.early_stop is not None:
            #     if trigger >= self.args.early_stop:
            #         print("=> early stopping")
            #         break
            if epoch >= self.args.warm_up:
                scheduler.step()
    
    def joint_train_on_source(self):
            
        print(f"Source Domain: {self.args.domain_source} Model: {self.args.name}")
        
        train_transform = A.Compose([
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.5),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.ColorJitter(brightness=0.6)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

        train_source_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain_source}/Train/*')
        val_paths = glob(f'/mnt/asgard2/data/lingkai/braTS20/{self.args.domain_source}/Test/*')

        train_source_dataset = BratsTrainContrastDataset(train_source_paths, augmentation=train_transform)
        val_dataset = BratsTestDataset(val_paths, augmentation=None)

        train_source_loader = torch.utils.data.DataLoader(train_source_dataset,batch_size=self.args.batch_size,shuffle=True,pin_memory=True,drop_last=False)

        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,pin_memory=True,drop_last=False)

        model = Unet_SimCLR(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
       
        model.apply(weights_init_kaiming) 
        
        criterion = BCEDiceLoss().to(device=self.device) # supervise loss function 

        optimizer = torch.optim.Adam(model.parameters(), self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0, last_epoch=-1) # Decrease the learning rate
        scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)
        
        start_time = time.time()
        end_time = time.time()
        best_val_loss = 10000
        best_val_dice = 0
        dice_avg_train = []
        val_dice_ = []
        loss_supervise_avg_ = []
        loss_contrast_avg_ = []
        compute_metric = Dice()
        trigger = 0
        model.train()
        for epoch in range(self.args.epochs):
            for i, data in tqdm(enumerate(train_source_loader, start=1), total=len(train_source_loader)):

                image1, image2, label1, label2 = data

                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    z1, predict1 = model(image1)
                    z2, _ = model(image2)

                    supervise_loss = criterion(predict1, label1)
 
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)

                # memory.push(z1=z1, z2=z2)     
                
                
                    contrast_loss = self.nt_xent_loss(z1, z2) / 2
                # dynamic lambda later
                    total_loss =  self.args.lam * supervise_loss + contrast_loss
                # backprop
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                compute_metric.update(predict=predict1, label=label1, loss_sup=supervise_loss, loss_con=contrast_loss)

            trigger += 1
            dice_avg, loss_supervise_avg, loss_contrast_avg = compute_metric.compute()
            dice_avg = torch.mean(dice_avg).item()
            loss_supervise_avg = loss_supervise_avg.item()
            loss_contrast_avg = loss_contrast_avg.item()
            dice_avg_train.append(dice_avg)
            loss_supervise_avg_.append(loss_supervise_avg)
            loss_contrast_avg_.append(loss_contrast_avg)
            print('Training: Lr: {} Epoch [{:0>3} / {:0>3}] Total Loss: {:.4f} Supervise Loss: {:.4f} Contrastive Loss {:.4f} Dice: {:.4f}'.format(
                optimizer.param_groups[0]['lr'], epoch + 1, self.args.epochs, loss_supervise_avg*self.args.lam + loss_contrast_avg, loss_supervise_avg, loss_contrast_avg, dice_avg
            ))
            
            compute_metric.reset()
            if epoch % self.args.validate_frequency == 0:
                start_time = time.time()
                val_dice = self.validate_dice(self.args, val_loader, model)
                val_dice_.append(val_dice)
                end_time = time.time()
                draw_training_joint_on_source(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_train, val_dice_, fig_name=self.args.name)
                if val_dice > best_val_dice:
                    trigger = 0
                    best_val_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(f'./models/{self.args.name}', 'best_dice_model.pt'))
                    print("Saving best model")
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Dice: {:.4f} Time: {:.2f}s \n".format(
                    epoch + 1, self.args.epochs, val_dice,
                    (end_time - start_time)))
            # early stopping
            if self.args.early_stop is not None:
                if trigger >= self.args.early_stop:
                    print("=> early stopping")
                    break
            if epoch >= self.args.warm_up:
                scheduler.step()
                
    def validate_contrastive(self, model, val_loader):
        model.eval()
        with torch.no_grad():
            avg_loss = 0
            counter = 0
            for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                image_val1, image_val2, _, _ = data
                image_val1 = image_val1.to(self.device)
                image_val2 = image_val2.to(self.device)
                with torch.no_grad():
                    con_loss = self.calculate_contrast_loss(model, image_val1, image_val2)
                avg_loss += con_loss
                
                counter += 1

            avg_loss /= counter
        model.train()
        return avg_loss
    
    def validate_dice(self, args, val_loader, model):
        diceMetric = Dice()
        segclr_model_dict = model.state_dict()
        Unet_model = Unet(in_channel=args.input_channel, out_channel=args.output_channel).to(self.device)
        Unet_model_dict = Unet_model.state_dict()
        segclr_model_dict = {k: v for k, v in segclr_model_dict.items() if k in Unet_model_dict}
        Unet_model_dict.update(segclr_model_dict)
        Unet_model.load_state_dict(Unet_model_dict)
        Unet_model.eval()
        with torch.no_grad():
            for batch_idx, (images, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
                predicted = torch.empty((3, 240, 240, 155))
                _, _, _, nums_z = images['flair'].shape
                for slice_num_z in range(0, nums_z):
                    flair_slice = images['flair'][:, :, :, slice_num_z]
                    t1ce_slice = images['t1ce'][:, :, :, slice_num_z]
                    t1_slice = images['t1'][:, :, :, slice_num_z]
                    t2_slice = images['t2'][:, :, :, slice_num_z]
                    images_slice = np.concatenate((flair_slice, t1_slice, t1ce_slice, t2_slice), axis=0)
                    images_slice = torch.from_numpy(np.expand_dims(images_slice, 0)).to(self.device)
                    output = Unet_model(images_slice)
                    predicted[:, :, :, slice_num_z] = output.cpu().detach()
                diceMetric.update(predicted.unsqueeze(0), label, torch.tensor(0), torch.tensor(0))
        dice_avg, _ , _ = diceMetric.compute()
        dice_wl, dice_tc, dice_et = dice_avg
        dice_avg = torch.mean(dice_avg).detach().numpy()
        print(f"Model: {args.domain_source} Domain: {args.domain_source} Average Dice: {dice_avg} Whole Tumor: {dice_wl} Tumor Core: {dice_tc} Enhanced Tumor: {dice_et}")
        return dice_avg

    # def validate(self, model, val_loader):
    #     with torch.no_grad():
    #         model.eval()
    #         val_loss = 0
    #         avg_dice = 0
    #         counter = 0
    #         for image1, image2, label1, label2 in val_loader:
    #             image1 = image1.to(self.device)
    #             image2 = image2.to(self.device)

    #             loss, dice = self.calculate_contrast_loss(model, image1, image2, label1, label2)

    #             val_loss += loss.item()
    #             avg_dice += dice
                
    #             counter += 1

    #         val_loss /= counter
    #         avg_dice /= counter
    #     model.train()
       

  # train_transform = A.Compose(
        # [
        # # A.Resize(200, 200),
        # # A.CropNonEmptyMaskIfExists(height=150, width=150),
        # A.HorizontalFlip(p=0.5),
        # A.Affine(scale=(1.0, 1.5), p=0.5),
        # A.Affine(translate_percent=(0, 0.25), p=0.5),
        # A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.5), 
        # A.GaussNoise(var_limit=(0, 0.33), p=0.15),
        # A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8)],
        # # A.RandomBrightness(limit=(0.7, 1.3), p=0.15)],
        # additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        # )   
        # train_transform = A.Compose(
        # [
        # # A.Resize(200, 200),
        # # A.CropNonEmptyMaskIfExists(height=150, width=150),
        # A.HorizontalFlip(p=0.5),
        # A.Affine(scale=(1.0, 1.5), p=0.5),
        # A.Affine(translate_percent=(0, 0.25), p=0.5),
        # A.ColorJitter(brightness=0.6, p=0.8), 
        # ],
        # additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        # )
        
            #   print(i)
            #     if i == 0:
            #         for batch_idx, source_data in tqdm(enumerate(train_source_dataset, start=1), total=len(train_source_dataset)):
            #             image_source1, image_source2, label_source1, label_source2 = source_data
            #             image_source1 = image_source1.to(self.device)
            #             image_source2 = image_source2.to(self.device)
            #             z1_source, predict1_source = model(image_source1)
            #             z2_source, _ = model(image_source2)
            #             supervise_loss_source = criterion(predict1_source, label_source1)
            #             contrast_loss_source = self.nt_xent_loss(z1_source.cpu(), z2.cpu())







