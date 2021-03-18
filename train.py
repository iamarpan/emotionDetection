import torch 
import torchvision
import torch.optim as optim
import torchvision.transforms as transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def extract_data(path,transform):
    dataset = ImageFolder(path,transform)
    return dataset


def display_images(images,labels):
    plt.imshow(np.transpose(images,(1,2,0)))
    plt.savefig("image.png")

if __name__ == '__main__':
    train_transform = transform.Compose(
            [   transform.CenterCrop(224),
                transform.ToTensor(),
                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                
                ])

    test_transform = transform.Compose(
            [   transform.CenterCrop(224),
                transform.ToTensor(),
                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ])

    train_set = extract_data("./images/train/",train_transform)
    val_set = extract_data("./images/validation",test_transform)
    train_loader = DataLoader(train_set,batch_size=16,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=16,shuffle=True)
    images,labels = next(iter(train_loader))
    images = torchvision.utils.make_grid(images)
    display_images(images,labels)
