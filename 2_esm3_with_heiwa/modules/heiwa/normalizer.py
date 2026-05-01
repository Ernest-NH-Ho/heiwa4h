import torch.nn as nn
import torch

class HEIWA_normalizer:
    def __init__(self):
        folder = "pca/structure,secondary_structure,sasa_structure_wifMulti_l-1"
        self.avg = torch.load(f"{folder}/avg1536_wt.pt", map_location="cuda", weights_only=True)
        self.std = torch.load(f"{folder}/sd1536_wt.pt", map_location="cuda", weights_only=True)
        self.eigvec = torch.load(f"{folder}/eigvec.pt", map_location="cuda", weights_only=True)

    def forward(self, x):
        x_out = (x-self.avg)/(self.std*1000)
        x_out = x_out @ (self.eigvec*2)
        return x_out

    def reverse(self, x):
        x_out = x @ torch.linalg.pinv(self.eigvec*2)
        x_out = (x_out * self.std*1000) + self.avg
        return x_out