import torch
from meta_test_loader import *
import torch.nn as nn
import sys
from copy import deepcopy
#sys.path.append('./torchFewShot/models/')
#from wrn_new import wrn
#from wrn_28 import Wide_ResNet
#from res_160 import ResNet
from resnet_drop import resnet12
from densenet import densenet121
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#from resnet12 import resnet12
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import OrderedDict
#from  tiered_loader import dataloader
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


def val_nn_with_drop(model, loader):
    model.eval()
    ave_acc = Averager() 
    ave_acc_o= Averager() 
    pick_list = [[1, 1, 0], [1, 0, 0], [0, 0, 0]]

    for i, batch in enumerate(loader, 1):
        choose_str = 'full'
        data =  batch['image']
        data = data.cuda()
        k = 5 *1
        data_shot, data_query = data[:k], data[k:]
        x = model(data_shot)
        x = x.squeeze().detach().cpu() 
        x_test = model(data_query)
        x_test = x_test.squeeze().detach().cpu()
        x_n = x / torch.norm(x,p=2,  dim=1, keepdim=True)
        x_n = x_n.reshape(1, 5, -1).mean(dim=0)
        p = x_n
        x_test_n = x_test /torch.norm(x_test,p=2,  dim=1, keepdim=True)
        logits = euclidean_metric(x_test_n , p)
        label = torch.arange(5).repeat(15)
        label = label.type(torch.LongTensor)
        acc = count_acc(logits, label)
        ave_acc_o.add(acc)
        #### random drop
        #data_shot, data_query = data[:k], data[k:]
        for k in range(len(pick_list)):
            x = model.forward_with_drop(data_shot, pick_list[k])
            x = x.squeeze().detach().cpu() 
            x_test = model.forward_with_drop(data_query, pick_list[k])
            x_test = x_test.squeeze().detach().cpu()
            x_n = x /torch.norm(x,p=2,  dim=1, keepdim=True)
            x_n = x_n.reshape(1, 5, -1).mean(dim=0)
            p = x_n
            x_test_n = x_test/torch.norm(x_test,p=2,  dim=1, keepdim=True)
            logits = euclidean_metric(x_test_n , p)
            label = torch.arange(5).repeat(15)
            label = label.type(torch.LongTensor)
            acc_ = count_acc(logits, label)
            if acc_ > acc:
                acc = acc_
                choose_str = pick_list[k]
                print(acc_ - acc)
        #print(choose_str)
        ave_acc.add(acc)
       # print(choose_str)
    print("One Shot Test Acc Original  Mean %.4f" % ( ave_acc_o.item()))
    print("One Shot Test Acc  Mean %.4f" % ( ave_acc.item()))

def val_nn_5(model, loader):
    model.eval()
    ave_acc = Averager()
    for i, batch in enumerate(loader, 1):
        data =  batch['image']
        data = data.cuda()
        k = 5 *5
        data_shot, data_query = data[:k], data[k:]
        x = model(data_shot)
        x = x.squeeze()
        x = x / torch.norm(x,p=2,  dim=1, keepdim=True)
        x = x.reshape(5, 5, -1).mean(dim=0)
        p = x
        x_test = model(data_query)
        x_test = x_test.squeeze()
        x_test = x_test / torch.norm(x_test,p=2,  dim=1, keepdim=True)
        logits = euclidean_metric(x_test , p)
        label = torch.arange(5).repeat(15)
        label = label.type(torch.cuda.LongTensor)
        acc = count_acc(logits, label)
        ave_acc.add(acc)
#        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        x = None; p = None; logits = None
    print("Five Shot Test Acc  Mean %.4f" % ( ave_acc.item()))


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
       # data =  batch['image']
       # data = data.cuda()
       # k=5
        train_targets =  torch.arange(5).repeat(1).type(torch.cuda.LongTensor)
        train_targets = train_targets.cpu().detach().numpy()
       # train_embeddings = model(train_inputs)
    #    train_embeddings = train_embeddings / torch.norm(train_embeddings,p=2,  dim=1, keepdim=True)
       # train_embeddings = train_embeddings.detach().cpu().numpy()
        head.fit(x_n, train_targets)
        test_targets = torch.arange(5).repeat(15).type(torch.cuda.LongTensor)
        test_targets = test_targets.cpu().detach().numpy()
        x_test_n = x_test
        #test_embeddings =  model(test_inputs)
     #   test_embeddings = test_embeddings / torch.norm(test_embeddings,p=2,  dim=1, keepdim=True)
      #  test_embeddings = test_embeddings.detach().cpu().numpy()
        test_pred = head.predict(x_test_n)
        acc = np.mean(test_pred == test_targets)
    #    print(acc)
     #   for k in range(100):
     #       pick_list = generate_list(ratio=0.5)
     #       x_ = deepcopy(x)
     #       x_test_ = deepcopy(x_test)
     #       x_[:,  pick_list] = 0
     #       x_ = x_ / torch.norm(x,p=2,  dim=1, keepdim=True)
     #       x_ = x_.reshape(1, 5, -1).mean(dim=0)
     #       p = x_
     #       x_test_[:,  pick_list] = 0
     #       x_test_ = x_test_ / torch.norm(x_test_, p=2,  dim=1, keepdim=True)
     #       head.fit(x_, train_targets)
     #       test_pred = head.predict(x_test_)
     #       acc_new = np.mean(test_pred == test_targets)
     #      # acc_new = count_acc(logits, label)
     #       if acc_new > acc:
     #           acc = acc_new
   #             print(acc)
        acc_list.append(acc)
    print("LR One Shot Val Acc %.4f" % (np.mean(acc_list)))


def val_lr_5(model, dataloader):
    model.eval()
    ################################# LR 
    print('LR')
    head = LogisticRegression(C=10, multi_class='multinomial', solver='lbfgs', max_iter=1000)
    acc_list = []
    for batch_idx, batch in enumerate(dataloader):
        data =  batch['image']
        data = data.cuda()
        k=5*5
        train_inputs, train_targets = data[:k], torch.arange(5).repeat(5).type(torch.cuda.LongTensor)
        train_targets = train_targets.cpu().detach().numpy()
        train_embeddings = model(train_inputs)
        train_embeddings = train_embeddings / torch.norm(train_embeddings,p=2,  dim=1, keepdim=True)
        train_embeddings = train_embeddings.detach().cpu().numpy()
        head.fit(train_embeddings, train_targets)
        test_inputs, test_targets = data[k :], torch.arange(5).repeat(15).type(torch.cuda.LongTensor)
        test_targets = test_targets.cpu().numpy()
        test_embeddings =  model(test_inputs)
        test_embeddings = test_embeddings / torch.norm(test_embeddings,p=2,  dim=1, keepdim=True)
        test_embeddings = test_embeddings.detach().cpu().numpy()
        test_pred = head.predict(test_embeddings)
        accuracy = np.mean(test_pred == test_targets)
        acc_list.append(accuracy)
    print("LR Five Shot Val Acc %.4f" % (np.mean(acc_list)))


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
       # data =  batch['image']
       # data = data.cuda()
       # k=5
        train_targets = torch.arange(5).repeat(1).type(torch.cuda.LongTensor)
        train_targets = train_targets.cpu().detach().numpy()
       # train_embeddings = model(train_inputs)
    #    train_embeddings = train_embeddings / torch.norm(train_embeddings,p=2,  dim=1, keepdim=True)
       # train_embeddings = train_embeddings.detach().cpu().numpy()
        head.fit(x_n, train_targets)
        test_targets = torch.arange(5).repeat(15).type(torch.cuda.LongTensor)
        test_targets = test_targets.cpu().detach().numpy()
        x_test_n = x_test 
        #test_embeddings =  model(test_inputs)
     #   test_embeddings = test_embeddings / torch.norm(test_embeddings,p=2,  dim=1, keepdim=True)
      #  test_embeddings = test_embeddings.detach().cpu().numpy()
        test_pred = head.predict(x_test_n)
        acc = np.mean(test_pred == test_targets)
     #   print(acc)
 #       for k in range(100):
 #           pick_list = generate_list(ratio=0.5)
 #           x_ = deepcopy(x)
 #           x_test_ = deepcopy(x_test)
 #           x_[:,  pick_list] = 0
 #           x_ = x_ / torch.norm(x,p=2,  dim=1, keepdim=True)
 #           x_ = x_.reshape(1, 5, -1).mean(dim=0)
 #           p = x_
 #           x_test_[:,  pick_list] = 0
 #           x_test_ = x_test_ / torch.norm(x_test_, p=2,  dim=1, keepdim=True)
 #           head.fit(x_, train_targets)
 #           test_pred = head.predict(x_test_)
 #           acc_new = np.mean(test_pred == test_targets)
         #   acc_new = count_acc(logits, label)
 #           if acc_new > acc:
 #               acc = acc_new
      #          print(acc)
        acc_list.append(acc)
    print("SVM One Shot Val Acc %.4f" % (np.mean(acc_list)))


def val_svm_5(model, dataloader):
    model.eval()
    ################################# LR 
    print('LR')
    head =  SVC(C=10, gamma='auto', kernel='linear')
    acc_list = []
    for batch_idx, batch in enumerate(dataloader):
        data =  batch['image']
        data = data.cuda()
        k=5*5
        train_inputs, train_targets = data[:k], torch.arange(5).repeat(5).type(torch.cuda.LongTensor)
        train_targets = train_targets.cpu().numpy()
        train_embeddings = model(train_inputs)
        train_embeddings = train_embeddings / torch.norm(train_embeddings,p=2,  dim=1, keepdim=True)
        train_embeddings = train_embeddings.detach().cpu().numpy()
        head.fit(train_embeddings, train_targets)
        test_inputs, test_targets = data[k :], torch.arange(5).repeat(15).type(torch.cuda.LongTensor)
        test_targets = test_targets.cpu().numpy()
        test_embeddings =  model(test_inputs)
        test_embeddings = test_embeddings / torch.norm(test_embeddings,p=2,  dim=1, keepdim=True)
        test_embeddings = test_embeddings.detach().cpu().numpy()
        test_pred = head.predict(test_embeddings)
        accuracy = np.mean(test_pred == test_targets)
        acc_list.append(accuracy)
    print("SVM Five Shot Val Acc %.4f" % (np.mean(acc_list)))

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
dataset = miniImageNet_test_dataset()
sampler = CategoriesSampler(dataset.label, 2000, 5, 16)
dataloader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
#sampler_5 = CategoriesSampler(dataset.label, 2000, 5, 20)
#dataloader_5 = DataLoader(dataset, batch_sampler=sampler_5, shuffle=False, num_workers=4, pin_memory=True)

## Res12
with torch.no_grad():
 #   model1 = densenet121().cuda()
    model1 = resnet12(meta_test=True).cuda()
    #model1 = nn.DataParallel(model1)
    model1.eval()
    model1.load_state_dict( torch.load('./res12_best.pth'))
    print('Resnet 12 NN')
#list1 = []
#for i in range(100):
#val_nn(model1, dataloader, 0.5)
#print(np.max(np.array(list1)))

#model1 = Wide_ResNet().cuda()
#model1.load_state_dict(remove_prefix( torch.load('./pretrain_weights/wrn_pre.pth')['params']),  strict=False)
#print('WRN NN')
#model1 = nn.DataParallel(model1)
    val_nn_with_drop(model1, dataloader)

#val_nn_5(model1, dataloader_5)
#print('Resnet 12 LR')
#val_lr(model1, dataloader)

#val_lr_5(model1, dataloader_5)
#print('Resnet 12 SVM')
#val_svm(model1, dataloader)
#val_svm_5(model1, dataloader_5)
ssss
## WRN 
#model2 = wr(meta_test=True)
#model1.load_state_dict( remove_prefix(torch.load('./pretrain_weights/MiniIMageNet-Res-1-Shot-5-Way.pth')['params']), strict=False)
#model2.cuda(
#model1.eval()
#print('WRN NN')
#val_nn(model1, dataloader)
#val_nn_5(model2, dataloader_5)
#print('WRN LR')
#val_lr(model2, dataloader)
#val_lr_5(model2, dataloader_5)
#print('WRN SVM')
#val_svm(model2, dataloader)
#val_svm_5(model2, dataloader_5)
