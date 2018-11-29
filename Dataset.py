import torch
from torch.utils import data
from utils.fs_utils import get_all_filenames
import os

class Dataset():
    def __init__(self, folderPath):

        self.files = get_all_filenames(folderPath, '**/*.png')
        print('loaded ', len(self.files), ' files')
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]
        X = torch.load(filename)

        path, filename = os.path.split(filename)

        pieces = filename.split('_')
        
        #parse the parameters from file name and return those too
        pieces = name.split('_')[1:] #split and throw images away

        # print(pieces)
        id = pieces[0]
        phi = pieces[1].strip('p')
        theta = pieces[2].strip('t')
        rho = pieces[3].strip('r').split('.')[0]

        return X, phi, theta, rho

if __name__ == '__main__':

    name = 'image_000_p001_t002_r003.png'
    pieces = name.split('_')[1:] #split and throw images away

    print(pieces)
    id = pieces[0]
    phi = pieces[1].strip('p')
    theta = pieces[2].strip('t')
    rho = pieces[3].strip('r').split('.')[0]
    
    print('id: ', id)
    print('theta: ', theta)
    print('phi: ', phi)
    print('rho: ', rho)

    for fullpath in get_all_filenames('//ark/E/datasets/rendered_chairs/', '**/*.png'):
        path, filename = os.path.split(fullpath)
        print(filename)