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
    
    with torch.no_grad():
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
        
        # add the batch axis
        c = c.unsqueeze(0)
        v = v.unsqueeze(0)
        t = t.unsqueeze(0)
    
        # compute out the image and mask
        image, mask = model(c, v, t)
        
        # convert to the form which can be displayed by plt.
        image = image.squeeze().numpy().transpose(1, 2, 0)
        image[np.where(image<=0)] = 0
        image[np.where(image>=1)] = 1
        mask = mask.squeeze().numpy()
        
        return image, mask

    
if __name__ == "__main__":
    
    best_model = "./best_model_78.pth"
    image, mask = import_best_model(best_model, 666, 20, 300)
    
    # create two subplots, show the image at the first
    # and the mask at the second one.
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()


