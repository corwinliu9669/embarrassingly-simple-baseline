import torch
from meta_test_loader import *
import torch.nn as nn
import sys
from copy import deepcopy
from resnet12 import resnet12
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import OrderedDict
def remove_prefix(weights_dict):
    w_d = OrderedDict()
    for k, v in weights_dict.items():
        new_k = k.replace('encoder.', '')
        print(new_k)
        w_d[new_k] = v
    return w_d

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.FloatTensor).mean().item()

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
def generate_list(ratio=0.5):
    pick_list =[]
    for k in range(512):
        if np.random.rand(1) <= ratio:
            pick_list.append(k)
    return pick_list
def val_nn(model, loader):
    model.eval()
    ave_acc = Averager() 
    for i, batch in enumerate(loader, 1):
        data =  batch['image']
        data = data.cuda()
        k = 5 *1
        data_shot, data_query = data[:k], data[k:]
        x = model(data_shot)
        x = x.squeeze().detach().cpu() 
        x_test = model(data_query)
        x_test = x_test.squeeze().detach().cpu()
        x_n = x 
        x_n = x_n.reshape(1, 5, -1).mean(dim=0)
        p = x_n
        x_test_n = x_test
        logits = euclidean_metric(x_test_n , p)
        label = torch.arange(5).repeat(15)
        label = label.type(torch.LongTensor)
        acc = count_acc(logits, label)
        ave_acc.add(acc)
    print("One Shot Test Acc  Mean %.4f" % ( ave_acc.item()))

def val_lr(model, dataloader):
    model.eval()
    ################################# LR 
    print('LR')
    head = LogisticRegression(C=10, multi_class='multinomial', solver='lbfgs', max_iter=1000)
    acc_list = []
    for batch_idx, batch in enumerate(dataloader):
        data =  batch['image']
        data = data.cuda()
        k = 5 *1
        data_shot, data_query = data[:k], data[k:]
        x = model(data_shot)
        x = x.squeeze().detach().cpu() 
        x_test = model(data_query)
        x_test = x_test.squeeze().detach().cpu()
        x_n = x 
        x_n = x_n.reshape(1, 5, -1).mean(dim=0)
        train_targets =  torch.arange(5).repeat(1).type(torch.cuda.LongTensor)
        train_targets = train_targets.cpu().detach().numpy())
        head.fit(x_n, train_targets)
        test_targets = torch.arange(5).repeat(15).type(torch.cuda.LongTensor)
        test_targets = test_targets.cpu().detach().numpy()
        x_test_n = x_test
        test_pred = head.predict(x_test_n)
        acc = np.mean(test_pred == test_targets)
        acc_list.append(acc)
    print("LR One Shot Val Acc %.4f" % (np.mean(acc_list)))



def val_svm(model, dataloader):
    model.eval()
    ################################# LR 
    print('LR')
    head =  SVC(C=10, gamma='auto', kernel='linear')
    acc_list = []
    for batch_idx, batch in enumerate(dataloader):
        data =  batch['image']
        data = data.cuda()
        k = 5 *1
        data_shot, data_query = data[:k], data[k:]
        x = model(data_shot)
        x = x.squeeze().detach().cpu() 
        x_test = model(data_query)
        x_test = x_test.squeeze().detach().cpu()
        x_n = x 
        x_n = x_n.reshape(1, 5, -1).mean(dim=0)
        train_targets = torch.arange(5).repeat(1).type(torch.cuda.LongTensor)
        train_targets = train_targets.cpu().detach().numpy()
        head.fit(x_n, train_targets)
        test_targets = torch.arange(5).repeat(15).type(torch.cuda.LongTensor)
        test_targets = test_targets.cpu().detach().numpy()
        x_test_n = x_test 
        test_pred = head.predict(x_test_n)
        acc = np.mean(test_pred == test_targets)
        acc_list.append(acc)
    print("SVM One Shot Val Acc %.4f" % (np.mean(acc_list)))

dataset = miniImageNet_test_dataset()
sampler = CategoriesSampler(dataset.label, 2000, 5, 16)
dataloader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)

## Res12
with torch.no_grad():
    model1 = resnet12(meta_test=True).cuda()
    model1.eval()
    model1.load_state_dict( torch.load('./res12_best.pth'))
    print('Resnet 12 NN')
    val_nn(model1, dataloader)


