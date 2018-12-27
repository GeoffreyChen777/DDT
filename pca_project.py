import torch.nn as nn
import torch
import numpy as np

class PCAProjectNet(nn.Module):
    def __init__(self):
        super(PCAProjectNet, self).__init__()

    def forward(self, features):     # features: NCWH
        k = features.size(0) * features.size(2) * features.size(3)
        x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        features = features - x_mean

        reshaped_features = features.view(features.size(0), features.size(1), -1)\
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)

        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        eigval, eigvec = torch.eig(cov, eigenvectors=True)

        first_compo = eigvec[:, 0]

        projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
            .view(features.size(0), features.size(2), features.size(3))
        return projected_map


if __name__ == '__main__':
    img = torch.randn(6, 512, 14, 14)
    pca = PCAProjectNet()
    pca(img)
