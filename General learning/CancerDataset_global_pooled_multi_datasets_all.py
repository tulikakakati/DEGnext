#DEGnext 
#Copyright Tulika Kakati, 2020 
#Distributed under GPL v3 license
#This code may be used to import cancer RNA-seq dataset(s) into Pytorch DataLoader

from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import Normalizer
import torch


class MyDataset(Dataset):
    """ My dataset."""

    # Initialize your data, download, etc.
    def __init__(self,data_type=None,case=None,transform=None):
        print(data_type)
        if(case=='train'):
            print(" single train dataset loading")
            File = './'+data_type+'_non_bio_train.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            File = './'+data_type+'_non_bio_train_label.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.float32)
            if(is_prime(xy.shape[1])==True):
                    M=xy.shape[0]
                    N=xy.shape[1]
                    b = np.zeros((M,N+1))
                    b[:,:-1] = xy
                    xy=b

            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            #print("new_data",xy.shape)
            xy=new_data
            self.transform=transform
            self.x_data=xy#[1:1000,]
            self.len=self.x_data.shape[0]
            self.y_data = xy_label#[1:1000,]



        if(case=='bio-fine-tune'):
            print("fine-tune-data loading")
            File = './'+data_type+'_bio_fine_tune.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            File = './'+data_type+'_bio_fine_tune_label.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.float32)
            if(is_prime(xy.shape[1])==True):
                    M=xy.shape[0]
                    N=xy.shape[1]
                    b = np.zeros((M,N+1))
                    b[:,:-1] = xy
                    xy=b

            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            #print("new_data",xy.shape)
            xy=new_data
            #print("gene exp before normalization:", xy.shape)
            #transformer = Normalizer().fit(xy)
            #xy=transformer.transform(xy)
            #print("gene exp after normalization:", xy.shape)

            self.transform=transform
            self.x_data=xy#[1:100,]
            self.len=self.x_data.shape[0]
            self.y_data = xy_label#[1:100,]


        if(case=='non-bio-test'):
            print(" non-bio-test dataset loading")
            File = './'+data_type+'_non_bio_test.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            File = './'+data_type+'_non_bio_test_label.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.float32)
            if(is_prime(xy.shape[1])==True):
                    M=xy.shape[0]
                    N=xy.shape[1]
                    b = np.zeros((M,N+1))
                    b[:,:-1] = xy
                    xy=b

            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            #print("new_data",xy.shape)
            xy=new_data
            self.transform=transform
            self.x_data=xy#[1:1000,]
            self.len=self.x_data.shape[0]
            self.y_data = xy_label#[1:1000,]

        if(case=='bio-test'):
            print(" bio test dataset loading")
            File = './'+data_type+'_bio_test.csv'
            xy = np.loadtxt(File,delimiter=',', dtype=np.float32)
            File = './'+data_type+'_bio_test_label.txt'
            xy_label = np.loadtxt(File,delimiter='\t', dtype=np.float32)
            if(is_prime(xy.shape[1])==True):
                    M=xy.shape[0]
                    N=xy.shape[1]
                    b = np.zeros((M,N+1))
                    b[:,:-1] = xy
                    xy=b

            N=xy.shape[0]
            noise_factor=1
            noise_data = np.random.randn(xy.shape[0],xy.shape[1])*noise_factor
            new_data=xy + noise_data
            #print("new_data",xy.shape)
            xy=new_data
            self.transform=transform
            self.x_data=xy#[1:1000,]
            self.len=self.x_data.shape[0]
            self.y_data = xy_label#[1:1000,]



    def __getitem__(self, index):
        #print("Index",index)
        return self.transform(self.x_data[index]),self.y_data[index]

    def __len__(self):
        return self.len

def is_prime(a):
    return all(a % i for i in range(2, a))



