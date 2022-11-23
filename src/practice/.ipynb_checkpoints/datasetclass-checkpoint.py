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

#import matplotlib


class MyDataset(Dataset):

    def __init__(self, annotation_file, img_dir,transform = None, target_transform=None):
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        label, corners = self.create_data()
        self.img_label = label
        self.corner_keypoints = corners
        self.transform = transform

    def create_data(self):
        label = []
        corners = []
        with open (os.path.join(self.img_dir, self.annotation_file)) as f:
            keypoints_data = json.load(f)
            keypoints_data = keypoints_data['dataset']
        for i in range(0,len(keypoints_data)):
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
        #sample = {'img' : img, 'corners' : corners}
        # if self.transform:
        #     img,target = self.transform(target)

        return img, target








def Display(train_dataloader) :

    train_img, train_target = next(iter(train_dataloader))


    fig, ax = plt.subplots(3,3,figsize=(9,9))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)


    ax = ax.ravel()


    for i in range(0,len(train_img)):

        img = train_img[i].squeeze()
        ax[i].imshow(img)
        corners = np.array(train_target['keypoints'][i])
        corners = np.multiply(corners,256)
        for j in range(0,len(corners)):
            corners[j][1] = 256-corners[j][1]
        P = patches.Polygon(corners, linewidth=1, edgecolor='r', facecolor='none')
        ax[i].add_patch(P)
        ax[i].set_title(train_target['labels'][i], fontsize=8)
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    plt.show()





if __name__=="__main__":
    DIR = r'C:\\Users\Administrator\Desktop\AIOR_Group\\Project1\src\box_dataset'
    dataobj = MyDataset(img_dir = DIR, annotation_file = 'dataset.json', transform= True)

    train_dataloader = DataLoader(dataobj, batch_size=9, shuffle=True)
    train_img, train_target = next(iter(train_dataloader))
    print(type(train_img),train_img.shape)
    print(type(train_target['keypoints']),train_target['keypoints'].shape)
    Display(train_dataloader)
    # print(dataobj[10])


#prepare data to feed Model






