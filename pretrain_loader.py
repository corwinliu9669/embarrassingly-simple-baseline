import os
import torch
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

list2np =  lambda x : np.array(x)
np2image = lambda x : Image.fromarray(x)

mini_transform_train= transforms.Compose([list2np,
                                     np2image,
                                     transforms.RandomCrop(84, padding=8),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ])
mini_transform_val= transforms.Compose([list2np,
                                     np2image,
                                     transforms.Resize(84),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ])
class miniImageNet_pretrain_dataset(Dataset):

    def __init__(self, pickle_file,  split='train'):
        """
        Args:
            pickle file (string): Name of pickle file we should load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.pickle_file = pickle_file
        self.data, self.label =  self.load_data(self.pickle_file) # load pickle file
        if split == 'train':
            self.transform = mini_transform_train
        elif split == 'val':
            self.transform = mini_transform_val
        else:
            pass

    def __len__(self):
        return len(self.data)
    
    def load_data(self, pickle_file):
        with open(pickle_file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data['data'], data['labels']

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_sample_raw = self.data[idx]
        image_sample = self.transform(image_sample_raw)
        label_sample = self.label[idx]
        sample = {'image': image_sample, 'label': label_sample}
        return sample


dataTrain = miniImageNet_pretrain_dataset(pickle_file = '/root/XCM/miniimagenet_pickle/miniImageNet_category_split_train_phase_train.pickle', split='train')
dataloader_train = torch.utils.data.DataLoader(dataTrain, batch_size=32, shuffle=True, num_workers=4)
dataVal = miniImageNet_pretrain_dataset(pickle_file = '/root/XCM/miniimagenet_pickle/miniImageNet_category_split_train_phase_val.pickle', split='val')
dataloader_val = torch.utils.data.DataLoader(dataVal, batch_size=32, shuffle=False, num_workers=4)
