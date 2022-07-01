import math
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from models import MyModel
from dataset import MLDataset
from torch.utils.data import DataLoader,random_split
import random
from matplotlib import pyplot as plt

# WRMSE


def WRMSE(preds, labels, device):
    weight = torch.tensor([
        0.05223, 0.0506, 0.05231, 0.05063, 0.05073,
        0.05227, 0.05177, 0.05186, 0.05076, 0.05063,
        0.0173, 0.05233, 0.05227, 0.05257, 0.05259,
        0.05222, 0.05204, 0.05185, 0.05229, 0.05074
    ]).to(device)
    wrmse = torch.pow(preds-labels, 2)
    wrmse = torch.sum(wrmse * weight)
    return wrmse.item()

# training curve


def visualize(record,v_record, title):
    plt.title(title)
    plt.plot(record)
    plt.plot(v_record)
    plt.show()

# learning rate, epoch and batch size. Can change the parameters here.


def train(lr=0.004, epoch=200, batch_size=32):
    train_loss_curve = []
    train_wrmse_curve = []
    validation_loss_curve = []
    validation_wrmse_curve = []
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel()
    model = model.to(device)
    model.train()

    # dataset and dataloader
    # can use torch random_split to create the validation dataset
    dataset = MLDataset()
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    train_set_size = int(len(dataset) * 0.9)
    valid_set_size = len(dataset)-train_set_size
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])
    trainloader = DataLoader(train_set,batch_size=batch_size)
    validloader = DataLoader(valid_set,batch_size=batch_size)

    # loss function and optimizer
    # can change loss function and optimizer you want
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    for e in range(epoch):
        train_loss = 0.0
        train_wrmse = 0.0
        best = 100
        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
        # tqdm to disply progress bar
        for inputs, labels in tqdm(trainloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)
            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss calculate
            train_loss += loss.item()
            train_wrmse += wrmse
        # =================================================================== #
        # If you have created the validation dataset,
        # you can refer to the for loop above and calculate the validation loss
        validation_loss = 0.0
        validation_wrmse = 0.0
        # tqdm to disply progress bar
        for inputs, labels in tqdm(validloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)
            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss calculate
            validation_loss += loss.item()
            validation_wrmse += wrmse
     


        # =================================================================== #
        # save the best model weights as .pth file
        loss_epoch = train_loss / len(trainloader.dataset)
        wrmse_epoch = math.sqrt(train_wrmse/len(trainloader.dataset))
        v_loss_epoch = validation_loss / len(validloader.dataset)
        v_wrmse_epoch = math.sqrt(validation_wrmse/len(validloader.dataset))
        if wrmse_epoch < best :
            best = wrmse_epoch
            torch.save(model.state_dict(), 'mymodel.pth')
        print(f'Training loss: {loss_epoch:.6f}')
        print(f'validation loss: {v_loss_epoch:.6f}')
        print(f'Training WRMSE: {wrmse_epoch:.6f}')
        print(f'validation WRMSE: {v_wrmse_epoch:.6f}')
        # save loss and wrmse every epoch
        train_loss_curve.append(loss_epoch)
        train_wrmse_curve.append(wrmse_epoch)
        validation_loss_curve.append(v_loss_epoch)
        validation_wrmse_curve.append(v_wrmse_epoch)
    # generate training curve
    visualize(train_loss_curve,validation_loss_curve, 'loss curve ')
    visualize(train_wrmse_curve,validation_wrmse_curve ,'WRMSE curve ')

if __name__ == '__main__':
    train()
