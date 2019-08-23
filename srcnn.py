import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib

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
        img = Image.open(img_name)

        if self.transform is not None:
            img = self.transform(img)

        return img


def load_images(root_dir):
    root_dir = pathlib.Path(root_dir)
    img_paths = list(root_dir.glob('**/*.*'))
    img_paths = [str(path) for path in img_paths]
    
    n_imgs = len(img_paths)
    print(n_imgs, 'image paths imported from', root_dir)

    dataset = ImageDataset(root_dir, img_paths)
    sample = dataset[0]
    
    plt.imshow(sample)
    plt.show();

'''
    transform = transforms.Compose)[
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),

    img = Image.open("")
    img_transformed = transform(img)
    batch = torch.unsqueeze()

    image_dataset = ImageDataset(root_dir=root_dir)

    for i in range(len(image_dataset)):
        sample = image_dataset[i]
'''


def train():
    return 0


def valid():
    return 0


def test():
    return 0


def main():
    load_images("dataset/")


if __name__ == '__main__':
    main()
