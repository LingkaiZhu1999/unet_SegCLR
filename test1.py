import os
from unet import Unet_SimCLR
import torch
from glob import glob
from dataset import BratsValidationDataset
from tqdm import tqdm
from metrics import dice_coef
import numpy as np
import pickle
torch.set_printoptions(threshold=10000)

def main():
    torch.cuda.manual_seed_all(1)
    model = Unet_SimCLR()
    model = model.cuda()

    # img_path = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_CBICA_HGG/image/*')
    # mask_path = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_CBICA_HGG/label/*')
    with open("valImgPath", "rb") as fp:
        img_path = pickle.load(fp)
        
    with open("valLabelPath", "rb") as fp:
        mask_path = pickle.load(fp)

    model.load_state_dict(torch.load('models/best_model.pt'))
    test_dataset = BratsValidationDataset(img_path, mask_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    whole_tumor_dice_ = []
    tumor_core_dice_ = []
    enhanced_tumor_dice_ = []
    dice_ = []

    model.eval()
    mlp1 = model.projector.projector_layer[1].weight
    mlp2 = model.projector.projector_layer[3].weight
    mlp3 = model.projector.projector_layer[5].weight
    with open("mlp3.txt", 'w') as file:
        file.write(str(mlp3))
    with torch.no_grad():
        for i, (image, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image = image.cuda()
            _, predict = model(image)
            # predict = (torch.sigmoid(predict).detach().cpu() > 0.5).int()
            whole_tumor_dice = dice_coef(predict[:, 0], target[:, 0])
            tumor_core_dice = dice_coef(predict[:, 1], target[:, 1])
            enhanced_tumor_dice = dice_coef(predict[:, 2], target[:, 2])
            
            dice = dice_coef(predict, target)
            dice_.append(dice)
            whole_tumor_dice_.append(whole_tumor_dice)
            tumor_core_dice_.append(tumor_core_dice)
            enhanced_tumor_dice_.append(enhanced_tumor_dice)
    whole_tumor_dice_avg = np.mean(whole_tumor_dice_)
    tumor_core_dice_avg = np.mean(tumor_core_dice_)
    enhanced_tumor_dice_avg = np.mean(enhanced_tumor_dice_)
    dice_avg1 = np.mean(dice_)
    dice_avg2 = (whole_tumor_dice_avg + tumor_core_dice_avg + enhanced_tumor_dice_avg) / 3
    print(f"whole_tumor_dice_avg: {whole_tumor_dice_avg}", f"tumor_core_dice_avg: {tumor_core_dice_avg}",
    f"enhanced_tumor_dice_avg: {enhanced_tumor_dice_avg}", f"dice_avg: {dice_avg1, dice_avg2}")

if __name__ == "__main__":
    main()




