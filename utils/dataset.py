import json
import os
import random
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

torch.manual_seed(0)


class segDataset(Dataset):
    "This is a wrapper for the seg dataset"

    def __init__(self, dataset_path, transforms=None):
        super(segDataset, self).__init__()
        if not os.path.exists(dataset_path):
            print("cannot find {}".format(dataset_path))
            exit(-1)
        self.dataset_path = os.path.abspath(dataset_path)
        self.imgList = []
        self.labelList = []
        with open(dataset_path) as f:
            lines = f.readlines()
        for line in lines:
            imgPath,labelPath = line.strip().split(",")
            self.imgList.append(os.path.join(os.path.dirname(self.dataset_path), imgPath))
            self.labelList.append(os.path.join(os.path.dirname(self.dataset_path), labelPath))
        self.length = len(self.imgList)
        self.transforms = transforms   

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        imgPath = self.imgList[idx]
        labelPath = self.labelList[idx]
        img = cv2.imread(imgPath,0)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        label = cv2.imread(labelPath, 0)
        label = label.astype(np.int64)
        return img,label