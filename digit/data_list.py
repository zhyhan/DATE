import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
import scipy.io as sio

class DigitFiveList(Dataset):

    def __init__(self, args, file_dir, labels=None, transform=None, target_transform=None,  mode='RGB'):
        self.args = args
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(file_dir)
        self.transform = transform
        #print(loaded_mat)
        if self.args.t ==0:
            self.data = loaded_mat['train_32']
            labels = loaded_mat['label_train'].astype(np.int64).squeeze()
            #one-hot to scalar
            self.labels = [] 
            for i in labels:
                for j, z in enumerate(i):
                    if z==1:
                        self.labels.append(int(j))
            #self.labels=
        #print(loaded_mat)
        if self.args.t == 1:
            self.data = loaded_mat['train']

            labels = loaded_mat['label_train'].astype(np.int64).squeeze()

            #one-hot to scalar
            self.labels = [] 
            for i in labels:
                for j, z in enumerate(i):
                    if z==1:
                        self.labels.append(int(j))

                #print(len(self.labels))
        if self.args.t == 2: 
            self.data = loaded_mat['X']
            self.labels = loaded_mat['y'].astype(np.int64).squeeze()
            self.data = np.transpose(self.data, (3, 2, 0, 1))
            np.place(self.labels, self.labels == 10, 0)

        if self.args.t == 3: 
            self.data = loaded_mat['X']
            self.labels = loaded_mat['y'].astype(np.int64).squeeze()
            self.data = np.transpose(self.data, (3, 2, 0, 1))
            #print(len(self.data), len(self.labels))
            np.place(self.labels, self.labels == 10, 0)

        if self.args.t == 4:
            #print(loaded_mat['dataset'])
            self.data = loaded_mat['dataset'][0][0].squeeze()
            #print(len(self.data))
            self.data *= 255.0
            self.data = np.squeeze(self.data).astype(np.uint8)
            labels = loaded_mat['dataset'][0][1].astype(np.int64).squeeze()
            #one-hot to scalar
            self.labels = labels
            #print(self.labels)
            # for i in labels:
            #     for j, z in enumerate(i):
            #         if z==1:
            #             self.labels.append(int(j))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(np.transpose(img, (1, 2, 0)))#TODO
        #img = Image.fromarray(img)
        if self.args.t == 0:
            #print(img.shape)
            img = Image.fromarray(img, mode='L')

        if self.args.t == 1:
            #print(img.shape)
            img = Image.fromarray(img)

        if self.args.t == 2:
            #print(img.shape)
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.args.t == 3:
            #print(img.shape)
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.args.t == 4:
            #print(img.shape)
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            #print(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        #print(len(self.data))
        return len(self.data)