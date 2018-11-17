import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # classes
        self.fc1_c = nn.Linear(809, 512)
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

        # add a reshape here

        # upsample layers
        # self.upconv1 = nn.Conv2d(8, 256, kernel_size=3)
        # self.conv1_1 = nn.Conv2d(256, 16, kernel_size=3)
        # self.upconv2 = nn.Conv2d(16, 256, kernel_size=3)
        # self.conv2_1 = nn.Conv2d(256, 16, kernel_size=3)
        # self.upconv3 = nn.Conv2d(16, 256, kernel_size=3)
        # self.conv3_1 = nn.Conv2d(, kernel_size=3)
        # self.upconv4 = nn.Conv2d(, kernel_size=3)

    def forward(self, c, v, t):

        # each layer add F.relu()
        c = self.fc1_c(c)
        c = self.fc2_c(c)

        v = self.fc1_v(v)
        v = self.fc2_v(v)

        t = self.fc1_t(t)
        t = self.fc2_t(t)

        x = torch.cat([c, v, t], dim=1)

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        

        return x
