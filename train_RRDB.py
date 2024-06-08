import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from dataloader import NCKUFinalDataset
from tqdm import tqdm
#from densenet_twolayer import *

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet_TWOLAYER(nn.Module):
    def __init__(self, growth_rate=32, num_classes=5):
        super(DenseNet_TWOLAYER, self).__init__()
        
        # first layer
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, growth_rate * 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(growth_rate * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # second layer
        self.dense_block1 = DenseBlock(growth_rate * 2, growth_rate, num_layers=1)
        
        
        self.transition1 = TransitionLayer(growth_rate * 3, growth_rate * 3)
        
        #self.dense_block2 = DenseBlock(growth_rate * 3, growth_rate, num_layers=4)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(growth_rate * 3, num_classes)

    def forward(self, x):
    
        #first
        
        x = self.initial_layer(x)
        
        # second
        x = self.dense_block1(x)
        
        x = self.transition1(x)
        
        #x = self.dense_block2(x)
        x = self.global_pool(x)
        
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x
        
        
        
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset_train = NCKUFinalDataset(txt_file="/ssd4/ming/final/train.txt", transform=data_transforms['train'])
dataset_val = NCKUFinalDataset(txt_file="/ssd4/ming/final/val.txt", transform=data_transforms['val'])

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=8)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=1)

model = DenseNet_TWOLAYER(num_classes=50)
#state_dict = torch.load("/ssd4/ming/final/best_rrdb_model.pth")
#model.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def adjust_lr(optimizer, num_epochs):
    if num_epochs in [100, 150, 180]:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
            lr = p['lr']
        print('Change lr:'+str(lr))
        
        
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    adjust_lr(optimizer, num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = dataloaders[phase]
            progress_bar = tqdm(dataloader, desc=f"{phase} {epoch}/{num_epochs - 1}")

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / len(labels))

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

dataloaders = {'train': dataloader_train, 'val': dataloader_val}
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=200)

torch.save(model.state_dict(), 'best_rrdb_model.pth')
