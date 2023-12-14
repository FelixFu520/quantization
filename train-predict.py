import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from utils import unet

torch.backends.cudnn.deterministic = True	
torch.backends.cudnn.benchmark = True	#False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run():
    PRETRAIN_MODEL = "./checkpoints/wafer-train.pth"
    # imgPath = "./data/wafer/crop/0000_Row006_Col036_00137_14.bmp"
    imgPath = "./data/wafer/crop/0500_Row023_Col050_00953_21.bmp"
    img = cv2.imread(imgPath,0)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = torch.from_numpy(img).to(device) #torch.Size([1, 1, 256, 256])
    model = unet(nclass=5)
    model.load_state_dict(torch.load(PRETRAIN_MODEL))
    model.to(device)
    model.eval()
    out = model(img)  #torch.Size([1, 5, 256, 256])
    pred = F.softmax(out, dim=1) #torch.Size([1, 5, 256, 256])
    pred = pred.detach().cpu().numpy()
    result=pred[0] #c, h, w
    seg_result_ori = np.argmax(result, axis=0).astype(np.uint8) #(512, 512)
    seg_result = (seg_result_ori != 0).astype(np.uint8)  # (h, w) # 目前先把所有前景类别当成一类显示
    seg_result_prob = seg_result * result.max(axis=0)
    cv2.imwrite("wafer-train-predict.png", seg_result_prob * 255)
    
if __name__ == "__main__":
    run()