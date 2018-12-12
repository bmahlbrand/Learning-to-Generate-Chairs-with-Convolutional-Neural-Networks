"""
 This script defines some functions used for visualize the final results.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from modules.model import Net

def import_best_model(best_model, category, phi, theta):
    """
    Argumnets:
            best_model: .pth file containing model parameters
            category: an integer 1 ~ 809
            phi: an integer 20 or 30
            theta: an integer 0 ~ 359 
    Outputs: returned image and mask by the pretrained model        
    """
    # load the best model
    state_dict = torch.load(best_model)
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    
    # convert c, phi and theta into corresponding inputs for the network
    # construct the class vector
    c = np.zeros(809)
    c[category-1] = 1
        
    # construct the view vector
    v = np.zeros(4)
    v[0] = np.sin(theta/180 * np.pi)
    v[1] = np.cos(theta/180 * np.pi)
    v[2] = np.sin(phi/180 * np.pi)
    v[3] = np.sin(phi/180 * np.pi)
        
    # construct the tranform parameter vector
    t = np.ones(12)
        
    # transform them into tensor and into tench.FloatTensor
    c = torch.from_numpy(c)
    v = torch.from_numpy(v)
    t = torch.from_numpy(t)
    c = c.float()
    v = v.float()
    t = t.float()
    
    # compute out the image and mask
    image, mask = model(c, v, t)

    return image, mask

def imshow(image, is_image=True):
    """
    Argument:
            image: image or mask, returned from import_best_model
            is_image: True for showing image, False for showing mask
    """
    if is_image:
        plt.imshow(image.numpy().transpose(1, 2, 0))
    else:
        plt.imshow(image.numpy())
    
    
if __name__ == "__main__":
    
    best_model = "./best_model_78.pth"
    image, mask = import_best_model(best_model, 666, 20, 300)
    imshow(image)

