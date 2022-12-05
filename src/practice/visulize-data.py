import json
import os

import cv2 as cv
import numpy as np
#import matplotlib

class create_data:

    def __init__(self, main_path, file_name):
        with open (os.path.join(main_path, file_name)) as f:
            self.keypoints_data = json.load(f)
            self.keypoints_data = self.keypoints_data['dataset']


        path, corners, img = self.create_data()
        self.img_path = path
        self.img = img                            #keypoints_data['image_path']
        self.corner_keypoints = corners                 #keypoints_data['corner_keypoints']
        #self.flap_corner_keypoints = keypoints_data['flap_corner_keypoints']
        #self.flap_center_keypoints = keypoints_data['flap_center_keypoints']
        self.train_data = list(zip(self.img[0:199], self.corner_keypoints[0:199]))
        self.val_data = list(zip(self.img[200:250], self.corner_keypoints[200:250]))


    def create_data(self):
        path = []
        cornerpoints = []
        img_store = []
        for i in range(0,len(self.keypoints_data)):
            dataT = self.keypoints_data[i]
            dataT2 = list(map(lambda j : j[::-1] , dataT))

            img = cv.imread(os.path.join(DIR,dataT['image_path']))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #print(img.shape)

            path.append(dataT['image_path'])
            img_store.append(img)
            corners = dataT['corner_keypoints']
            #corners2 = list(map(lambda j : j[::-1] , corners))
            cornerpoints.append(corners)
        return  path, cornerpoints, img_store

    def Normilize(self):
        pass





    def visulize(self, DIR, color = (0,255,0), thickness=2):
        x = np.random.randint(low=0, high=len(self.img_path))
        img = cv.imread(os.path.join(DIR,self.img_path[x]))
        pts = np.array(self.corner_keypoints[x], np.float32)
        pts = np.multiply(pts,256)
        for i in range(0,len(pts)):
            pts[i][1] = 256-pts[i][1]

        pts = pts.reshape((-1,1,2))

        print (pts)


        pts = pts.astype(int)
        print (pts)


        cv.polylines(img,[pts],True,color,thickness = thickness)
        cv.putText(img, str(self.img_path[x]).split('/')[1].split('.')[0],(0,30),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0), thickness=2)
        cv.imshow('sample1', img)
        print(self.corner_keypoints[x])
        cv.waitKey(0)


if __name__=="__main__":
    DIR = r'C:\\Users\Administrator\Desktop\AIOR_Group\\Project1\src\box_dataset'
    dataobj = create_data(main_path = DIR, file_name = 'dataset.json')
    dataobj.visulize(DIR)


