import torch
import torch.nn.functional as F
from torchvision import transforms
import random

class DiffAugment():
    
    def __init__(self,policy='',p=0.0):
        self.p = p
        self.transform_list = []
        if p > 0:
            for p in policy.split(','):
                self.transform_list.append(AUGMENT_FNS[p])

    def __call__(self,x):
        for transform in self.transform_list:
            if random.uniform(0,1) < self.p:
                x = transform(x)
        return x
        

AUGMENT_FNS = {
    'color': transforms.ColorJitter(),
    'randaffine': transforms.RandomAffine([-30,30],(0.05, 0.05),
                                                      (0.8, 1),
                                                      (-15, 15))
} 
