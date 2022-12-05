import cv2 as cv
import numpy as np
import torch
from torchvision import transforms, utils


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""
    def __call__(self,sample):

        image, key_pts, labels = sample['image'], sample['keypoints'], sample['labels']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        # convert image to grayscale
        #image_copy = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # scale color range from [0, 256] to [0, 1]
        image_copy=  image_copy/255.0


        # change the base coordinate to top left
        for j in range(0,len(key_pts_copy)):
                key_pts_copy[j][1] = 1-key_pts_copy[j][1]



        return {'image': image_copy, 'keypoints': key_pts_copy, 'labels' : labels }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        image, key_pts, labels = sample['image'], sample['keypoints'], sample['labels']
        #print(type(image))
        #print(image.shape)

        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #print(type(image))
        #print(image.shape)

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts), 'labels' : labels }

# class resize(object):
#     def __init__ (self, outputsize):
#         self.outputsize = outputsize
        
#     def __call__(self, sample):
#         image, key_pts, labels = sample['image'], sample['keypoints'], sample['labels']
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
            
#         new_h, new_w = int(new_h), int(new_w)
#         img = cv2.resize(image, (new_w, new_h))
#         key_pts = key_pts * [new_w / w, new_h / h]
#     return {'image': img, 'keypoints': key_pts}

data_transform = transforms.Compose([Normalize(), ToTensor()])
