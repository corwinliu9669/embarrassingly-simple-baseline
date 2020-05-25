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
class miniImageNet_test_dataset(Dataset):

    def __init__(self, pickle_file='/root/miniimagenet_pickle/miniImageNet_category_split_test.pickle',  split='val'):
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
            label_tmp = data['labels']
            label_min = min(label_tmp)
            label = [i-label_min for i in label_tmp]    
        return data['data'], label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_sample_raw = self.data[idx]
        image_sample = self.transform(image_sample_raw)
        label_sample = self.label[idx]
        sample = {'image': image_sample, 'label': label_sample}
        return sample



class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

#dataset = miniImageNet_test_dataset()
#sampler = CategoriesSampler(dataset.label, 2000, 5, 16)
#dataloader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
