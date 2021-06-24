import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.misc import toimage
import torch
import os
import matplotlib.image as img
import skimage
import scipy

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
        for idx, i in enumerate(listOfFiles[:100]):
            l = read_spectrogram(listOfFiles[idx])
            #print(l.shape)
            if idx % 200 == 0:
                print(idx)
            if idx == 0:
                concat = l.transpose(0,3,1,2)
            else:
                concat = np.concatenate((concat, l.transpose(0,3,1,2)))
            #concat.extend(l)
            #concat.extend(l)
        print(concat.shape)
        concat = concat.astype('float32')'''
        with open('./data/ISMIR/ismir_latest_spec_array.pkl', 'rb') as f:
            concat = pickle.load(f, encoding='bytes')
        concat =  np.array(list(concat.values()))
        print(concat.shape)
        concat = concat.astype('float32')
        #print(len(images))
        #print(images[0].shape)
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
        skimage.io.imread(spectrogram_path)[:, :, :3], (1,32, 32, 3)
        )

def read_spectrogram_batch(spectrogram_paths):
    # remove alpha dimension and resize to 224x224
    return [skimage.transform.resize(s[:, :, :3], (1, 32, 32, 3))
            for s in skimage.io.imread_collection(spectrogram_paths)]


def preprocess_data(train_data, test_data):
    return train_data.reshape((len(train_data), 3, 32, 32))/255.0,  test_data.reshape((len(test_data), 3, 32, 32))/255.0#.transpose(0,2,3,1)

train_data, test_data = get_data('Spectrograms')
print(train_data.shape)
print(len(train_data[2]))
a = torch.from_numpy(train_data[2])
print(a.shape)
a = a.permute(1,2,0)
print(a.shape)
a=a.numpy()
#plt.imshow(toimage(train_data[2]))
plt.imshow(a, interpolation='nearest')
plt.show()