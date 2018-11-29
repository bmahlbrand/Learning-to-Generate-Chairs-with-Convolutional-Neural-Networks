import torch
from torch.utils import data
from utils.fs_utils import get_all_filenames

class Dataset(data.Dataset):
  def __init__(self, folderPath):

    self.files = get_all_filenames(folderPath, '/*.png')
    print(self.files)

  def __len__(self):
        return len(self.files)

  def __getitem__(self, index):

        filename = self.files[index]

        
        X = torch.load('datasets/' + filename)
        #parse the parameters from file name and return those too

        return X
