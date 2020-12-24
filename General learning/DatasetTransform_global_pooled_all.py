#DEGnext 
#Copyright Tulika Kakati, 2020 
#Distributed under GPL v3 license
#This code may be used to transform cancer RNA-seq dataset(s)

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch
import math

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, gene_exp):
        #print('before',type(gene_exp))
        for transform in self.transforms:
            gene_exp= transform(gene_exp)


        return gene_exp

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to
        a list of torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, gene_exp):
        #a=gene_exp.shape[0]
        #print(gene_exp.shape[0])

        if(is_prime(gene_exp.shape[0])==True):
            print("Prime number")
        a = gene_exp.shape[0]
        a= math.sqrt(a)
        a=np.floor(a)
        a=a.astype(np.int)
        if(a*a!=gene_exp.shape[0]):
            for i in range(a):
                if(gene_exp.shape[0]%a!=0):
                    a=a-1

        gene_exp_new = np.reshape(gene_exp, (1,-1, a))

        gene_exp_new=torch.from_numpy(gene_exp_new).float()
        return gene_exp_new




    def __getitem__(self, index):
        return self.transform(self.x_data[index]),self.y_data[index]

    def __len__(self):
        return self.len

def is_prime(a):
    return all(a % i for i in range(2, a))
