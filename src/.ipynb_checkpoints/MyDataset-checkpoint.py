import json
import os

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.io import read_image

import transform

#import matplotlib


class MyDataset(Dataset):

    def __init__(self, annotation_file, img_dir,transform = transform, target_transform=None, datatype = 'train'):
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.datatype = datatype
        label, corners = self.read_data()
        self.img_label = label
        self.corner_keypoints = corners
        self.transform = transform
        

    def read_data(self):
        label = []
        corners = []
        with open (os.path.join(self.img_dir, self.annotation_file)) as f:
            keypoints_data = json.load(f)
            keypoints_data = keypoints_data['dataset']

        if self.datatype == 'train':
            portion = range(0,200)
            
            
        else:
            portion = range(200,250)
           

        for i in portion :
            
            dataT = keypoints_data[i]
            label.append(dataT['image_path'])
            corners.append(dataT['corner_keypoints'])
        label = list(map(lambda i : (i.split('/')[1]) ,label))
        return  label, corners


    def __len__(self):
            return len(self.img_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,'images', self.img_label[idx] )
        #image = read_image(img_path)
        img = cv.imread(img_path)
        label = self.img_label[idx]
        corners = self.corner_keypoints[idx]

        target = {}
        target["keypoints"] = torch.as_tensor(corners, dtype=torch.float32)
        target["labels"] =label
        target["image"] = img

        if self.transform:
            target = self.transform(target)

        return target



def Display(train_dataloader) :

    train_target = next(iter(train_dataloader))


    fig, ax = plt.subplots(3,3,figsize=(9,9))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)


    ax = ax.ravel()


    for i in range(0,len(train_target['image'])):

        img = train_target['image'][i].squeeze()
        print(img)
        print(img.shape)
        ax[i].imshow(img)
        corners = np.array(train_target['keypoints'][i])
        corners = np.multiply(corners,256)
        print(corners)
        P = patches.Polygon(corners, linewidth=1, edgecolor='r', facecolor='none')
        ax[i].add_patch(P)
        ax[i].set_title(train_target['labels'][i], fontsize=8)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    plt.show()





if __name__=="__main__":
    DIR = r'C:\\Users\Administrator\Desktop\AIOR_Group\\Project1\src\box_dataset'
    triandataobj = MyDataset(img_dir = DIR, annotation_file = 'dataset.json', transform = transform.data_transform, datatype = 'train')
    testdataobj = MyDataset(img_dir = DIR, annotation_file = 'dataset.json', transform = transform.data_transform, datatype = 'test')
    print(len(triandataobj.img_label))
    print(len(testdataobj.img_label))
    testdataobj[10]
    train_dataloader = DataLoader(triandataobj, batch_size=9, shuffle=True)
    test_dataloader = DataLoader(testdataobj, batch_size=9, shuffle=True)

    # train_target = next(iter(train_dataloader))
    # test_target = next(iter(test_dataloader))

    # print(len(test_target['labels']))
    # print(len(train_target['labels']))
    # # train_size = int(0.8 * len(train_dataloader))
    # test_size = len(train_dataloader) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(train_dataloader, [train_size, test_size])
    # train_target = next(iter(train_dataloader))
    Display(train_dataloader)
    # print(dataobj[10]['image'].shape)
    # # print(train_target['image'][0].shape)

    # i=0

    # for data in test_dataloader :

    #     i = i+1

    #     print (len(data['image']))
    # print (i)

#prepare data to feed Model






