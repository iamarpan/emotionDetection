import torch 
import torchvision
import torch.optim as optim
import torchvision.transforms as transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

def extract_data(path,transform):
    dataset = ImageFolder(path,transform)
    return dataset

def prepare_model():
    model = models.resnet101(pretrained=True) 
    model.fc.out_features=7
    for name,param in model.named_parameters():
        if(param.requires_grad==True and name not in ['fc.weight','fc.bias']):
            param.requires_grad = False

    for name,param in model.named_parameters():
        if(param.requires_grad==True):
            print(name)

    return model
    

def train(model,loader):
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    for _ in tqdm(range(10)):
        for images,labels in loader:
            output = model(images)
            optimizer.zero_grad()
            loss = F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()
        loss_array.append(loss.item())
    plt.plot(range(10),loss_array)
    plt.savefig("epoch/loss")

def display_images(images,labels):
    images = images*0.5 +0.5
    plt.imshow(np.transpose(images,(1,2,0)))
    plt.savefig("image.png")

if __name__ == '__main__':
    train_transform = transform.Compose(
            [   transform.RandomResizedCrop(224),
                transform.ToTensor(),
                transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                
                ])

    test_transform = transform.Compose(
            [   transform.RandomResizedCrop(224),
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
    model = prepare_model()
    train(model,train_loader)
