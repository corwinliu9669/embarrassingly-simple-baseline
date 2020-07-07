from pretrain_loader import dataloader_train, dataloader_val
from utils import *
import torch
import torch.nn as nn
import sys
from resnet12 import resnet12
import torchvision
from torch.optim import SGD
from tqdm import tqdm
import argparse
from meta_test_loader import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser('Pretrain Network')
parser.add_argument('--weights_folder', type=str, default='./pretrain_weights/',
        help='folder to store weights')
parser.add_argument('--epoch', type=int, default=100,
        help='Number of Epochs.')
parser.add_argument('--step', type=int, default=30,
        help='step to decay lr')
parser.add_argument('--use_cuda', default=True,
        help='Use CUDA if available.')
parser.add_argument('--lr_initial', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--txt_folder', type=str, default='./')
parser.add_argument('--optimizer', type=str, default='sgd')
args = parser.parse_args()
args.device = torch.device('cuda' if args.use_cuda
        and torch.cuda.is_available() else 'cpu')



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
        x_n = x / torch.norm(x,p=2,  dim=1, keepdim=True)
        x_n = x_n.reshape(1, 5, -1).mean(dim=0)
        p = x_n
        x_test_n = x_test / torch.norm(x_test,p=2,  dim=1, keepdim=True)
        logits = euclidean_metric(x_test_n , p)
        label = torch.arange(5).repeat(15)
        label = label.type(torch.LongTensor)
        acc = count_acc(logits, label)
        ave_acc.add(acc)
    print("One Shot Test Acc  Mean %.4f" % ( ave_acc.item()))
    return ave_acc.item()
def main(args):
##################################################################################################################
    #######  train parts
    # Model
    best_importance = 0
    record_file = open('resnet_vanilla.txt', 'w')
    model = densenet121()
    model.to(device=args.device)
    model.train()
    optimizer = SGD(model.parameters(), lr=args.lr_initial, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    dataloader, valloader = dataloader_train, dataloader_val
    best_acc = 0
    dataset = miniImageNet_test_dataset(pickle_file='/root/XCM/miniimagenet_pickle/miniImageNet_category_split_val.pickle',  split='val')
    #adataset = miniImageNet_test_dataset()
    sampler = CategoriesSampler(dataset.label, 600, 5, 16)
    metaloader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
    importance = val_nn(model,metaloader)
    for e in range(args.epoch):
        descent_lr(args.lr_initial, e, optimizer, 30)
        model.train()
        with tqdm(dataloader, total=64*600//64) as pbar:
            for batch_idx, batch in enumerate(pbar):
                model.zero_grad()
                train_inputs, train_targets = batch['image'], batch['label']
                train_inputs = train_inputs.to(device=args.device)
                train_targets = train_targets.to(device=args.device)
                train_logits = model(train_inputs)
                loss = nn.CrossEntropyLoss()(train_logits, train_targets)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    pbar.set_postfix(loss='{0:.4f}'.format(loss.item()))

        record_file.write('Epoch :  ')
        record_file.write(str(e+1))
        print('Validation')
        record_file.write('Validation:')
        record_file.write('\n')
        importance = val_nn(model, metaloader)
         #Save model
        if importance > best_importance:
            best_importance = importance
            filename = args.weights_folder  + 'dense_best.pth'
            state_dict = model.state_dict()
            torch.save(state_dict, filename)
    record_file.close()


main(args)
