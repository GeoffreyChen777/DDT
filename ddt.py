from .pca_project import PCAProjectNet
import os
from .vgg import *
from PIL import Image
import torchvision.transforms as tvt
import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import cv2
import numpy as np

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

data_list = os.listdir('./data')

model = vgg19(pretrained=True)

imgs = []

for name in data_list:
    img = image_trans(Image.open(os.path.join('./data'+name)).convert('RGB'))
    imgs.append(img.unsqueeze(0))

imgs = torch.cat(imgs)

features, _ = model(imgs)
pca = PCAProjectNet()

project_map = torch.clamp(pca(features), min=0)

maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
project_map /= maxv

project_map = F.interpolate(project_map.unsqueeze(1), size=(imgs.size(2), imgs.size(3)), mode='bilinear', align_corners=False) * 255.

save_imgs = []

for i, name in enumerate(data_list):
    img = cv2.resize(cv2.imread(os.path.join('./data', name)), (224, 224))
    mask = project_map[i].repeat(3, 1, 1).permute(1, 2, 0).detach().numpy()
    mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    save_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
    save_imgs.append(save_img)

save_imgs = np.concatenate(save_imgs, 1)
cv2.imwrite('./test.jpg', save_imgs)
