from model_2d_unet import UNet
from dataset import BratsTrainDataset
from torch.utils.data import DataLoader
import torch
from loss import Loss
from torch.nn import init

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Epoch = 1000

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)

def train():
    train_dataset = BratsTrainDataset(datapath='/mnt/asgard2/data/lingkai/braTS20/BraTS2020_TrainingData', augmentation=None)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, shuffle=True)
    model = UNet().to(device)
    model.apply(weights_init_kaiming)
    lossDice = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0, last_epoch=-1)
    model.train()
    num = 0
    for epoch in range(Epoch):
        trainloss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = lossDice(output, target)

            loss.backward()

            optimizer.step()

            trainloss = trainloss + loss
            if (batch_idx + 1) % 5 == 0:
                trainloss = trainloss / 5
                print('train Epoch {} iteration: {} loss: {:.6f}'.format(epoch, batch_idx + 1, trainloss.data))
                trainloss = 0

            if epoch >=250:
                scheduler.step()


train()
        



