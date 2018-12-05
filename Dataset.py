import os
import re
from numpy import newaxis
import torch
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F
from utils.fs_utils import get_all_filenames
from skimage import io
from matplotlib import pyplot as plt

class Dataset():
    def __init__(self, folderPath, output_size=128):
        """
         folderPath: like "../data/rendered_chairs"
         output_size: the desired output size of images and masks, in the paper, it is 64, 128, or 256. 
        """
        # get the list of image and mask filenames
        self.images = get_all_filenames(folderPath, '**/**/*.png')[0:809]
        self.masks = [re.sub(r'render', r'mask', image) for image in self.images]
        print('loaded ', len(self.images), ' files')
        # print(self.files)
        
        # do the transformation
        self.transformations = transforms.Compose([transforms.Resize(output_size),\
                                                  transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # load images
        image_filename = self.images[index]
        image = torch.from_numpy(io.imread(image_filename).transpose((2, 0, 1)))
        #print(image.size())
        #image = transforms.ToPILImage(image)
        image = F.to_pil_image(image)
        image = self.transformations(image)
        
        # load masks
        mask_filename = self.masks[index]
        #print(io.imread(mask_filename).shape)
        mask = torch.from_numpy(io.imread(mask_filename)[:,:,newaxis].transpose((2, 0, 1)))
        #print(mask.size())
        mask = F.to_pil_image(mask)
        mask = self.transformations(mask)

        # path: like "../data/rendered_chairs/b663f1e4df6c51fe19fb4103277a6b93/renders"
        # filename: like "image_001_p020_t011_r096.png"
        _, filename = os.path.split(image_filename)
        
        # get the pieces of image filename to parse the parameters
        # pieces = filename.split('_')
        
        #parse the parameters from file name and return those too
        pieces = filename.split('_')[1:] #split and throw images away

        # print(pieces)
        # id = pieces[0]
        # remove the first char to get the parameters
        phi = int(pieces[1].strip('p'))
        theta = int(pieces[2].strip('t'))
        rho = int(pieces[3].strip('r').split('.')[0])

        return image, mask, phi, theta, rho

if __name__ == '__main__':

    image, mask, phi, theta, rho = Dataset('../data/rendered_chairs/').__getitem__(100)
    print(image.size())
    print(mask.size())
    print(image.numpy().shape)
    print(mask.numpy().shape)
    #plt.imshow(image.numpy().transpose(1, 2, 0))
    plt.imshow(torch.squeeze(mask).numpy())
    plt.show()
    print("phi: ", phi)
    print("theta: ", theta)
    print("rho: ", rho)
    