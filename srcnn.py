import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math

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

        downsample = transforms.Compose([
            transforms.Resize(64, Image.BICUBIC),
            transforms.Resize(128, Image.BICUBIC),
            transforms.ToTensor()])

        lowres = downsample(gt)
        gt = transforms.Compose([transforms.ToTensor()])(gt)

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
                                  shuffle=False, num_workers=1)
    
    print(len(test_dataset), 'test image paths imported from', root_dir)

    return test_dataloader


def train(dataloader):
    device = torch.device('cuda')

    n_epochs = 30
    
    model = torch.nn.Sequential(
        torch.nn.ZeroPad2d(4),
        torch.nn.Conv2d(3, 64, (9,9)),
        torch.nn.ReLU(),
        
        torch.nn.ZeroPad2d(1),
        torch.nn.Conv2d(64, 32, (3,3)),
        torch.nn.ReLU(),
        
        torch.nn.ZeroPad2d(2),
        torch.nn.Conv2d(32, 3, (5,5))
    )
    
    model.to(device)

    loss_fn = torch.nn.MSELoss().to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        avg_loss = 0.0
        for i_batch, batch in enumerate(dataloader):
            x = batch['lowres'].to(device)
            y = batch['label'].to(device)

            y_pred = model(x)

            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss
            print('Epoch {}({}/{}) - Loss:{:.6f}'.format(epoch, 
                        i_batch, len(dataloader), loss.item()))
        
        avg_loss /= len(dataloader)
        avg_psnr = 10 * math.log10(1 / avg_loss)
        print('Epoch {} average - Loss:{:.6f} PSNR:{}'.format(
                    epoch, avg_loss, avg_psnr))

        save_model(model, epoch)
    
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


def use(model_path):
    device = torch.device('cuda')


def save_model(model, epoch):
    path = 'model_{}.pth'.format(epoch)
    torch.save(model, path)
    print(path, 'saved')


def main():
    train_dataloader = load_dataset('dataset/')
    test_dataloader = load_testset('testset/')

    model = train(train_dataloader)
    test(model, test_dataloader)


if __name__ == '__main__':
    main()
