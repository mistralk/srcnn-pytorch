import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math
from datetime import datetime

class ImageDataset(Dataset):
    def __init__(self, root_dir, path_list, transform=None):
        self.root_dir = root_dir
        self.path_list = path_list
        self.transform = transform

        self.data_len = len(path_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_name = self.path_list[idx]
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        gt = img
        
        lowres = img.filter(ImageFilter.GaussianBlur)
        downsample = transforms.Compose([
            transforms.Resize(64, Image.BICUBIC),
            transforms.Resize(128, Image.BICUBIC),
            transforms.ToTensor()])
        lowres = downsample(lowres)

        gt = transforms.Compose([
            transforms.CenterCrop(128-8-2-4),
            transforms.ToTensor()])(gt)

        return {'lowres': lowres, 'label': gt}


# Return the torch dataloader of the dataset
def load_dataset(root_dir):
    root_dir = pathlib.Path(root_dir)
    img_paths = list(root_dir.glob('**/*.*'))
    img_paths = [str(path) for path in img_paths]
    
    transform = transforms.Compose([
        transforms.CenterCrop(400),
        transforms.RandomCrop(128)])
    
    train_dataset = ImageDataset(root_dir, img_paths, transform)  
    train_dataloader = DataLoader(train_dataset, batch_size=128, 
                                  shuffle=True, num_workers=12)
    
    print(len(train_dataset), 'image paths imported from', root_dir)

    return train_dataloader

'''
    for i, sample in enumerate(dataloader):
        print(i)
        plt.imshow(sample.numpy()[i].transpose())
        plt.show();

        if i == 3:
            break
'''

def load_testset(root_dir):
    root_dir = pathlib.Path(root_dir)
    img_paths = list(root_dir.glob('**/*.*'))
    img_paths = [str(path) for path in img_paths]
    
    transform = transforms.Compose([
        transforms.CenterCrop(128)])
    
    test_dataset = ImageDataset(root_dir, img_paths, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), 
                                  shuffle=False, num_workers=12)
    
    print(len(test_dataset), 'test image paths imported from', root_dir)

    return test_dataloader


def train(dataloader, resume=False):
    device = torch.device('cuda')
    writer = SummaryWriter('runs/Aug25_19-22-24_mistralk-A320M-S2H/',purge_step=72830)

    n_epochs = 1000
    loss_fn = torch.nn.MSELoss().to(device)
    learning_rate = 0.001
    
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, (9,9)),
        torch.nn.ReLU(),

        torch.nn.Conv2d(64, 32, (3,3)),
        torch.nn.ReLU(),
        
        torch.nn.Conv2d(32, 3, (5,5))
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for g in optimizer.param_groups:
        g['lr'] = learning_rate

    start_epoch = 0
    
    if resume is True:
        model, optimizer, start_epoch = load_checkpoint(
                    'model_epoch_182.pth', model, optimizer, resume_training=True)

    for epoch in range(start_epoch, n_epochs):
        avg_loss = 0.0
        avg_psnr = 0.0
        for i_batch, batch in enumerate(dataloader):
            x = batch['lowres'].to(device)
            y = batch['label'].to(device)

            y_pred = model(x)
            y_pred = torch.clamp(y_pred, 0.0, 1.0)

            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            psnr = 10 * math.log10(1/loss)
            
            avg_loss += loss.item()
            avg_psnr += psnr
            print('Epoch {}({}/{}) - Loss:{:.6f} PSNR:{}'.format(epoch, 
                        i_batch, len(dataloader), loss.item(), psnr))

            step = epoch * len(dataloader) + i_batch
            if i_batch % 10 == 0:
                writer.add_scalar('Accuracy/MSE', loss.item(), step)
                writer.add_scalar('Accuracy/PSNR', psnr, step)
             
            if i_batch % 100 == 0:
                grid_x = utils.make_grid(x)
                grid_pred = utils.make_grid(y_pred)
                grid_y = utils.make_grid(y)
                
                writer.add_image('Result/lowres', grid_x, step)
                writer.add_image('Result/prediction', grid_pred, step)
                writer.add_image('Result/ground truth', grid_y, step)
            
        avg_loss /= len(dataloader)
        avg_psnr /= len(dataloader)
        print('Epoch {} average - Loss:{:.6f} PSNR:{}'.format(
                    epoch, avg_loss, avg_psnr))

        save_checkpoint(model, optimizer, epoch)
    
    writer.close()

    return model


def valid():
    return 0


def test(model, dataloader):
    device = torch.device('cuda')
    loss_fn = torch.nn.MSELoss().to(device)
    avg_psnr = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['lowres'].to(device)
            y = batch['label'].to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            psnr = 10 * math.log10(1 / loss.item())
            avg_psnr += psnr

    print('Test set - Average PSNR:{}'.format(avg_psnr))


def save_checkpoint(model, optimizer, epoch):
    path = 'model_epoch_{}.pth'.format(epoch)
    torch.save({'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()},
                path)
    print(path, 'saved')


def load_checkpoint(path, model, optimizer, resume_training=False):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1

    if resume_training is True:
        model.train()
    else:
        model.eval()

    return model, optimizer, epoch


def main():
    train_dataloader = load_dataset('dataset/')
    test_dataloader = load_testset('testset/')

    # Timer start 
    time_fmt = '%H:%M:%S'
    present = datetime.now().strftime(time_fmt)

    model = train(train_dataloader, resume=True)
    test(model, test_dataloader)

    # Timer end
    now = datetime.now().strftime(time_fmt)
    elapsed_time = datetime.strptime(now, time_fmt) - datetime.strptime(present, time_fmt)
    print()
    print('Elapsed time for training: ', elapsed_time)
    print()

if __name__ == '__main__':
    main()
