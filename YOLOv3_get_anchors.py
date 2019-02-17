#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:43:39 2019

@author: deniz
"""

import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()  # for plot styling

os.chdir("/media/deniz/02B89600B895F301/BBD100K")
train_path = "data/labels/train/bdd100k_labels_images_train.json"

with open(train_path,"r") as ftr:
    trlabel = json.load(ftr)
    
BBDlabeldict = {"bike":0,
             "bus":1,
             "car":2,
             "motor":3,
             "person":4,
             "rider":5,
             "traffic light":6,
             "traffic sign":7,
             "train":8,
             "truck":9,
             "drivable area":[],
             "lane":[]}

w,h = [] , []
for ind1 in range(len(trlabel)):
    for ind2 in range(len(trlabel[ind1]["labels"])):
        try:
            a=trlabel[ind1]["labels"][ind2]["box2d"]   #traffic sign
            x1,y1,x2,y2 = list(a.values())
            width = abs(x1-x2)
            height = abs(y1-y2)
            w.append(width)
            h.append(height)
        except:
            pass
w=np.asarray(w)
h=np.asarray(h)
     
x=[w,h]
x=np.asarray(x)
x=x.transpose()
##########################################   K- Means
##########################################

from sklearn.cluster import KMeans
kmeans3 = KMeans(n_clusters=9)
kmeans3.fit(x)
y_kmeans3 = kmeans3.predict(x)

##########################################
centers3 = kmeans3.cluster_centers_
#yolo_anchor_boxes3 = centers3
#yolo_anchor_boxes3[:,0] = centers3[:,0]/1280*608
#yolo_anchor_boxes3[:,1] = centers3[:,1]/720*608
#print(yolo_anchor_boxes3)
yolo_anchor_average=[]
for ind in range (9):
    yolo_anchor_average.append(np.mean(x[y_kmeans3==ind],axis=0))
    print((yolo_anchor_average[ind])/ 1280 * 608)

yolo_anchor_average=np.array(yolo_anchor_average)
print((yolo_anchor_average))

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=50, cmap='viridis')
#plt.scatter(centers3[:, 0], centers3[:, 1], c='black', s=200, alpha=0.5);
plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=200, alpha=0.5);
yolo_anchor_average.sort(axis=0)
yolo_anchor_average.sort(axis=1)

fig, ax = plt.subplots()
for ind in range(9):
#    rectangle= plt.Rectangle((304-yolo_anchor_boxes3[ind,0]/2,304-yolo_anchor_boxes3[ind,1]/2), yolo_anchor_boxes3[ind,0],yolo_anchor_boxes3[ind,1] , fc='r',edgecolor='r',fill = None)
#    ax.add_patch(rectangle)
    rectangle= plt.Rectangle((304-yolo_anchor_average[ind,0]/2,304-yolo_anchor_average[ind,1]/2), yolo_anchor_average[ind,0],yolo_anchor_average[ind,1] , fc='b',edgecolor='b',fill = None)
    ax.add_patch(rectangle)
ax.set_aspect(1.0)
plt.axis([0,608,0,608])
plt.show()

print("Your custom anchor boxes are {}".format(np.rint(yolo_anchor_average)))
