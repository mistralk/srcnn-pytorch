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
            gt = self.transform(img)

        downsample = transforms.Compose([
            transforms.Resize(112, Image.LANCZOS),
            transforms.Resize(224, Image.LANCZOS),
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
        transforms.CenterCrop(224)])
    
    train_dataset = ImageDataset(root_dir, img_paths, transform)  
    train_dataloader = DataLoader(train_dataset, batch_size=64, 
                                  shuffle=True, num_workers=6)
    
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

def train(dataloader):
    device = torch.device('cuda')

    n_epochs = 10
    
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
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        for i_batch, batch in enumerate(dataloader):
            x = batch['lowres'].to(device)
            y = batch['label'].to(device)

            y_pred = model(x)

            optimizer.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
    
            psnr = 10 * math.log10(1 / loss.item())

            print('Epoch {}({}/{}) - Loss:{:.6f} PSNR:{:.6f}'.format(epoch, 
                        i_batch, len(dataloader), loss.item(), psnr))

    return 0


def valid():
    return 0


def test():
    return 0


def main():
    train_dataloader = load_dataset('dataset/')
    train(train_dataloader)
    

if __name__ == '__main__':
    main()
