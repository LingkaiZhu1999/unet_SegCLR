import torch
from loss.nt_xent import NTXentLoss
import torch.nn.functional as F
from loss.BCEDiceLoss import BCEDiceLoss
from unet import Unet_SimCLR
from tqdm import tqdm
import time
from metrics import compute_dice, Dice
from dataset import BratsTrainDataset, BratsValidationDataset
from sklearn.model_selection import train_test_split
from glob import glob
import albumentations as A
import os
import numpy as np
from utils import draw_training
import pickle

class SimCLR(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0')
        self.nt_xent_loss = NTXentLoss('cpu', self.args.batch_size, self.args.temperature, use_cosine_similarity=True)

    def calculate_contrast_loss(self, model, x1, x2, label1, label2):
        z1, predict1 = model(x1)
        z2, predict2 = model(x2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)        
        contrast_loss = self.nt_xent_loss(z1.cpu(), z2.cpu())

        diceValue1 = compute_dice(predict1, label1)
        diceValue2 = compute_dice(predict2, label2)
        avg_dice = (diceValue1 + diceValue2) / 2 

        return contrast_loss, avg_dice

    def train(self):
        train_transform = A.Compose(
        [
        # A.Resize(200, 200),
        # A.CropNonEmptyMaskIfExists(height=150, width=150),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.15),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.15), 
        A.GaussNoise(var_limit=(0, 0.33), p=0.15),
        A.RandomBrightness(limit=(0.7, 1.3), p=0.15),
        A.RandomContrast(limit=(0.65, 1.5), p=0.15)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
        )

        with open("trainImgPath", "rb") as fp:
            trainImg_paths = pickle.load(fp)
        
        with open("trainLabelPath", "rb") as fp:
            trainMask_paths = pickle.load(fp)

        # trainImg_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_TCIA_HGG/image/*')
        # trainMask_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_TCIA_HGG/label/*')
        valImg_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_CBICA_HGG/image/*')
        valMask_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_CBICA_HGG/label/*')

        # train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        # train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
        train_dataset = BratsTrainDataset(trainImg_paths, trainMask_paths, augmentation=train_transform)
        val_dataset = BratsValidationDataset(valImg_paths, valMask_paths, augmentation=None)

        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.args.batch_size,shuffle=True,pin_memory=True,drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,pin_memory=True,drop_last=False)

        model = Unet_SimCLR(in_channel=self.args.input_channel, out_channel=self.args.output_channel).to(self.device)
        
        criterion = BCEDiceLoss() # supervise loss function 

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=0,
                                                               last_epoch=-1)

        start_time = time.time()
        end_time = time.time()
        best_val_loss = 10000
        dice_avg_ = []
        loss_supervise_avg_ = []
        loss_contrast_avg_ = []
        compute_metric = Dice()
        for epoch in range(self.args.epochs):
            for i, data in tqdm(enumerate(train_loader, start=1), total=len(train_loader)):

                image1, image2, label1, label2 = data

                image1 = image1.to(self.device)
                image2 = image2.to(self.device)

                z1, predict1 = model(image1)
                z2, predict2 = model(image2)

                supervise_loss1 = criterion(predict1.cpu(), label1.cpu())
                supervise_loss2 = criterion(predict2.cpu(), label2.cpu())

                supervise_loss = (supervise_loss1 + supervise_loss2) / 2


                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)        
                contrast_loss = self.nt_xent_loss(z1.cpu(), z2.cpu()) / 2
                # contrast_loss = self.calculate_contrast_loss(model, image1, image2)

                total_loss = 5*supervise_loss + contrast_loss 
                
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

                compute_metric.update(predict=predict1, label=label1, loss_sup=supervise_loss1, loss_con=contrast_loss)
                compute_metric.update(predict=predict2, label=label2, loss_sup=supervise_loss2, loss_con=contrast_loss)
                # if i % self.args.print_interval == 0:
                #     start_time, end_time = end_time, time.time()
                #     print("\nTraining:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ConLoss: {:.4f} Dice: {:.4f} Time: {:.2f}s".format(
                #         epoch + 1, self.args.epochs, i, len(train_loader), total_loss, contrast_loss, avg_dice,  end_time - start_time))
            dice_avg, loss_supervise_avg, loss_contrast_avg = compute_metric.compute()
            dice_avg = torch.mean(dice_avg).item()
            loss_supervise_avg = loss_supervise_avg.item()
            loss_contrast_avg = loss_contrast_avg.item()
            dice_avg_.append(dice_avg)
            loss_supervise_avg_.append(loss_supervise_avg)
            loss_contrast_avg_.append(loss_contrast_avg)
            print('Training: Epoch [{:0>3} / {:0>3}] Total Loss: {:.4f} Supervise Loss: {:.4f} Contrastive Loss {:.4f} Dice: {:.4f}'.format(
                epoch + 1, self.args.epochs, loss_supervise_avg + loss_contrast_avg, loss_supervise_avg, loss_contrast_avg, dice_avg
            ))
            
            draw_training(epoch + 1, loss_supervise_avg_, loss_contrast_avg_, dice_avg_)
            compute_metric.reset()
            if epoch % self.args.validate_frequency == 0:
                start_time = time.time()
                val_loss, val_dice = self.validate(model, val_loader, criterion)
                end_time = time.time()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join('./models', 'best_model.pt'))
                    print("Saving best model")
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Loss: {:.4f} Dice: {:.4f} Time: {:.2f}s".format(
                    epoch + 1, self.args.epochs, val_loss, val_dice, 
                    end_time - start_time))
            # if epoch >= 10:
            #     scheduler.step()
                
    def validate(self, model, val_loader, criterion):
        model.eval()
        with torch.no_grad():
            avg_dice = 0
            avg_loss = 0
            counter = 0
            for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                image, label = data
                image = image.to(self.device)
                with torch.no_grad():
                    _, predict = model(image)
                dice = compute_dice(predict, label)
                loss = criterion(predict.cpu(), label.cpu())
                avg_dice += dice
                avg_loss += loss
                
                counter += 1

            avg_dice /= counter
            avg_loss /= counter
        model.train()
        return avg_loss, avg_dice

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
       










