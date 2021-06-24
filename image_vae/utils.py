import numpy as np
import pickle
import torch
from torch.utils import data
from config import model_config as config
from model.vae import *
import os
import matplotlib.image as img
import skimage.transform
import skimage.io
import scipy


device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

class Dataset(data.Dataset):
    #'Characterizes a dataset for PyTorch'
    def __init__(self, x_pair):
        #'Initialization'
        self.x_train = x_pair

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.x_train)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample

        # Load data and get label
        x = self.x_train[index]

        return x

def get_data(dataset):
    if dataset == 'CIFAR10':
        with open('./data/CIFAR10/data_batch_1', 'rb') as f:
            t_1 = pickle.load(f, encoding='bytes')
            t_1 = t_1[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_2', 'rb') as f:
            t_2 = pickle.load(f, encoding='bytes')
            t_2 = t_2[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_3', 'rb') as f:
            t_3 = pickle.load(f, encoding='bytes')
            t_3 = t_3[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_4', 'rb') as f:
            t_4 = pickle.load(f, encoding='bytes')
            t_4 = t_4[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_5', 'rb') as f:
            t_5 = pickle.load(f, encoding='bytes')
            t_5 = t_5[b'data'] # 10000x3072
        f.close()
        train_data = np.concatenate((t_1, t_2, t_3, t_4, t_5))
        with open('./data/CIFAR10/test_batch', 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
            test_data = test_data[b'data']
        train_data, test_data = preprocess_data(train_data, test_data)
        train_data = train_data.astype('float32') #10000,3,32,32
        test_data = test_data.astype('float32')
        return train_data, test_data
    elif dataset == 'Spectrograms':
        '''dirName = './data/Lyrics/DepecheMode'
        # Get the list of all files in directory tree at given path
        listOfFiles = getListOfFiles(dirName)
        concat= []
        for idx, i in enumerate(listOfFiles[:32]):
            l = read_spectrogram(listOfFiles[idx])
            #print(l.shape)
            if idx % 2 == 0:
                print(idx)
            if idx == 0:
                concat = l.transpose(0,3,1,2)
            else:
                concat = np.concatenate((concat, l.transpose(0,3,1,2)))
            #concat.extend(l)
            #concat.extend(l)
        print(concat.shape)
        concat = concat.astype('float32')
        with open('data.pkl', 'wb') as f:
            pickle.dump(concat, f)'''
        #print(len(images))
        #print(images[0].shape)'''
        '''with open('data.pkl', 'rb') as f:
            concat = pickle.load(f, encoding='bytes')
        concat = concat.astype('float32')'''
        with open('./data/ISMIR/ismir_latest_spec_array.pkl', 'rb') as f:
            concat = pickle.load(f, encoding='bytes')
        concat =  np.array(list(concat.values()))
        print(concat.shape)
        concat = concat/255.0
        concat = concat.astype('float32')
        concat = concat.transpose(0,3,1,2)
        return concat, []
        #print(images.shape)

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles        


def read_spectrogram(spectrogram_path):
    # remove alpha dimension and resize to 224x224
    return skimage.transform.resize(
        skimage.io.imread(spectrogram_path)[:, :, :3], (1,224, 224, 3)
        )
def read_spectrogram_batch(spectrogram_paths):
    # remove alpha dimension and resize to 224x224
    return [skimage.transform.resize(s[:, :, :3], (1, 32, 32, 3))
            for s in skimage.io.imread_collection(spectrogram_paths)]

def preprocess_data(train_data, test_data):
    return train_data.reshape((len(train_data), 3, 32, 32))/255.0,  test_data.reshape((len(test_data), 3, 32, 32))/255.0#.transpose(0,2,3,1)

def get_a_sample(dataset):
    if dataset == 'CIFAR10':
        with open('./data/CIFAR10/data_batch_1', 'rb') as f:
            t_1 = pickle.load(f, encoding='bytes')
            t_1 = t_1[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_2', 'rb') as f:
            t_2 = pickle.load(f, encoding='bytes')
            t_2 = t_2[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_3', 'rb') as f:
            t_3 = pickle.load(f, encoding='bytes')
            t_3 = t_3[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_4', 'rb') as f:
            t_4 = pickle.load(f, encoding='bytes')
            t_4 = t_4[b'data'] # 10000x3072
        f.close()
        with open('./data/CIFAR10/data_batch_5', 'rb') as f:
            t_5 = pickle.load(f, encoding='bytes')
            t_5 = t_5[b'data'] # 10000x3072
        f.close()
        train_data = np.concatenate((t_1, t_2, t_3, t_4, t_5))
        train_data = train_data.reshape((len(train_data), 3, 32, 32))/255.0
        #save_image(torch.from_numpy(train_data[2]), 'sample_image1.png')
        train_data = train_data.astype('float32')
        a = torch.from_numpy(train_data[2])
        a = a.unsqueeze(0)
        print(a.shape)
        for i in range(a.size(0)):
            save_image(a[i, :, :, :], '{}.png'.format(i))
        #save_image(torch.from_numpy(train_data[2]), 'sample_image2.png')
        return train_data[2]
    elif dataset == 'Spectrograms':
        '''dirName = './data/Lyrics/DepecheMode'
        # Get the list of all files in directory tree at given path
        listOfFiles = getListOfFiles(dirName)
        concat= []
        for idx, i in enumerate(listOfFiles[:32]):
            l = read_spectrogram(listOfFiles[idx])
            #print(l.shape)
            if idx % 2 == 0:
                print(idx)
            if idx == 0:
                concat = l.transpose(0,3,1,2)
            else:
                concat = np.concatenate((concat, l.transpose(0,3,1,2)))
            #concat.extend(l)
            #concat.extend(l)
        print(concat.shape)
        concat = concat.astype('float32')
        with open('data.pkl', 'wb') as f:
            pickle.dump(concat, f)'''
        #print(len(images))
        #print(images[0].shape)'''
        '''with open('data.pkl', 'rb') as f:
            concat = pickle.load(f, encoding='bytes')
        concat = concat.astype('float32')'''
        with open('./data/ISMIR/ismir_latest_spec_array.pkl', 'rb') as f:
            concat = pickle.load(f, encoding='bytes')
        concat =  np.array(list(concat.values()))
        print(concat.shape)
        concat = concat/255.0
        concat = concat.astype('float32')
        concat = concat.transpose(0,3,1,2)
        return concat[1]
        #print(images.shape)

def load_data(dataset, batch_size):
    dataset = Dataset(dataset)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader