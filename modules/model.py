import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # paper used a dataset of 809 cleaned up classes
    def __init__(self, classes = 809):
        super(Net, self).__init__()

        # classes
        self.fc1_c = nn.Linear(classes, 512)
        self.fc2_c = nn.Linear(512, 512)

        # views
        self.fc1_v = nn.Linear(4, 512)
        self.fc2_v = nn.Linear(512, 512)

        # transforms
        self.fc1_t = nn.Linear(12, 512)
        self.fc2_t = nn.Linear(512, 512)

        self.fc3 = nn.Linear(1536, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 16384)

        # reshaping is in the forward function  
        
        # upsample layers
        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4,stride=2,padding=1)
        self.conv1_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 92, kernel_size=4, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(92, 92, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(92, 48, kernel_size=4, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        # upconv4 for generating the target color image
        self.upconv4_image = nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1)
        # upconv4 for generating the target segmentation mask
        self.upconv4_mask = nn.ConvTranspose2d(48, 1, kernel_size=4, stride=2, padding=1)
        
        # softmax for mask output if we use Cross-Entropy
        # self.softmax = nn.Softmax2d()
        # when use softmax, all the loss is 0, so we change sigmoid
        # and the loss function to nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, c, v, t):

        # process each input separately
        c = F.relu(self.fc1_c(c))
        c = F.relu(self.fc2_c(c))

        v = F.relu(self.fc1_v(v))
        v = F.relu(self.fc2_v(v))

        t = F.relu(self.fc1_t(t))
        t = F.relu(self.fc2_t(t))
        
        # concatenate three tensors
        x = torch.cat((c, v, t), dim=1)

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        # resize the 1-d tensor into 2-d tensor
        x = x.view(-1, 256, 8, 8)

        # use CNN to process 
        x = F.relu(self.upconv1(x))
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv3_1(x))

        # to get the two ouputs
        image = self.upconv4_image(x)
        # mask = self.upconv4_mask(x) # if use squared Euclidean distance
        # mask = self.softmax(self.upconv4_mask(x))  # if use NLL loss
        mask = self.sigmoid(self.upconv4_mask(x))
        return image, mask