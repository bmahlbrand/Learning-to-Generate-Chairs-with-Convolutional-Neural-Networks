# torch
import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as transforms

import numpy as np

# project
from utils.config_utils import load_config
from utils.Timer import Timer
from utils.AverageMeter import AverageMeter

from modules.model import Net
from Dataset import Dataset

from utils import viz_utils
from utils import torch_utils
from utils.fs_utils import create_folder

# system
import argparse
import time

create_folder('checkpoints')
folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)

create_folder('log')
logPath = 'log/log_' + Timer.timeFilenameString()

def append_line_to_log(line = '\n'):
    with open(logPath, 'a') as f:
        f.write(line + '\n')

torch.set_default_tensor_type('torch.cuda.FloatTensor')

params = load_config('config.yaml')

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
losses_list = [AverageMeter() for i in range(6)]
end = time.time()
best_model = params['best_model']

def parse_cli():
    parser = argparse.ArgumentParser(description='PyTorch ConvPoseMachines')

    parser.add_argument('--batch-size', type=int, default=params['batch_size'], metavar='N',
                        help='input batch size for training (default: ' + str(params['batch_size']) + ')')
    parser.add_argument('--epochs', type=int, default=params['epochs'], metavar='N',
                        help='number of epochs to train (default: ' + str(params['epochs']) + ')')

    ## hyperparameters
    parser.add_argument('--lr', type=float, default=params['init_learning_rate'], metavar='LR',
                        help='inital learning rate (default: ' + str(params['init_learning_rate']) + ')')

    parser.add_argument('--decay', type=float, default=params['decay'], metavar='DE',
                        help='SGD learning rate decay (default: ' + str(params['decay']) + ')')

    parser.add_argument('--beta1', type=float, default=params['beta1'], metavar='B1',
                        help=' Adam parameter beta1 (default: ' + str(params['beta1']) + ')')

    parser.add_argument('--beta2', type=float, default=params['beta2'], metavar='B2',
                        help=' Adam parameter beta2 (default: ' + str(params['beta2']) + ')')                    
                        
    parser.add_argument('--epsilon', type=float, default=params['epsilon'], metavar='EL',
                        help=' Adam regularization parameter (default: ' + str(params['epsilon']) + ')')

    parser.add_argument('--dampening', type=float, default=params['dampening'], metavar='DA',
                        help='SGD dampening (default: ' + str(params['dampening']) + ')')

    parser.add_argument('--seed', type=int, default=params['seed'], metavar='S',
                        help='random seed (default: ' + str(params['seed']) + ')')

    ## system training
    parser.add_argument('--log-interval', type=int, default=params['log_interval'], metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--workers', type=int, default=0, metavar='W',
                        help='workers (default: 0)')

    parser.add_argument('--train_dir', default='../data', type=str, metavar='PATHT',
                        help='path to latest checkpoint (default: data folder)')

    parser.add_argument('--val_dir', default='../data', type=str, metavar='PATHV',
                        help='path to latest checkpoint (default: data folder)')                    

    args = parser.parse_args()

    return args

def train(epoch, model, optimizer, criterion1, criterion2, lamb, loader, device, log_callback):
    
    end = time.time()
    model.train()

    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']

    # the output of the dataloader is (batch_idx, image, mask, c, v, t)
    for batch_idx, data in enumerate(loader):
        target_image, target_mask, input_c, input_v, input_t = data
        target_image = target_image.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        input_c = input_c.to(device, non_blocking=True)
        input_v = input_v.to(device, non_blocking=True)
        input_t = input_t.to(device, non_blocking=True)

        
        data_time.update(time.time() - end)
        
        # input all the input vectors into the model 
        out_image, out_mask = model(input_c, input_v, input_t)

        # compute the loss according to the paper
        loss1 = criterion1(out_image, target_image)
        # Note that the target should remove the channel size as stated in document
        #loss2 = criterion2(out_mask, target_mask.long().squeeze()) 
        loss2 = criterion2(out_mask, target_mask)
        loss = loss1 + lamb * loss2 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
 
        # record essential informations into log file.
        if batch_idx % args.log_interval == 0:
            log_callback('Epoch: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1} batches, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1} batches, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'.format(
                epoch, args.log_interval, learning_rate, batch_time=batch_time,
                data_time=data_time))
            
            log_callback('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
            log_callback()
            
            log_callback('Loss{0} = {loss1:.8f}\t'
                    .format(1, loss1=loss1.item()))
            
            log_callback('Loss{0} = {loss1:.8f}\t'
                    .format(2, loss1=loss2.item()))

            log_callback()
            log_callback("current time: " + Timer.timeString())
            
            batch_time.reset()
            data_time.reset()

    torch_utils.save(folderPath + 'ChairCNN_' + str(epoch) + '.cpkt', epoch, model, optimizer, scheduler)

def validation(model, criterion1, criterion2, lamb, loader, device, log_callback):
    end = time.time()
    model.eval()

    # return validation_loss, validation_acc
    with torch.no_grad():
        # the output of the dataloader is (batch_idx, image, mask, c, v, t)
        for batch_idx, data in enumerate(loader):
            target_image, target_mask, input_c, input_v, input_t = data
            target_image = target_image.to(device, non_blocking=True)
            target_mask = target_mask.to(device, non_blocking=True)
            input_c = input_c.to(device, non_blocking=True)
            input_v = input_v.to(device, non_blocking=True)
            input_t = input_t.to(device, non_blocking=True)

            # compute the output
            out_image, out_mask = model(input_c, input_v, input_t)
           
            # compute the loss
            loss1 = criterion1(out_image, target_image)
            #loss2 = criterion2(out_mask, target_mask.long().squeeze())
            loss2 = criterion2(out_mask, target_mask)
            loss = loss1 + lamb * loss2 
        
            batch_time.update(time.time() - end)
            end = time.time()

        # records essential information into log file.
        log_callback('epoch: {0}\t'
                'Time {batch_time.sum:.3f}s / {1} epochs, ({batch_time.avg:.3f})\t'
                'Data load {data_time.sum:.3f}s / {1} epochs, ({data_time.avg:3f})\n'
                'Loss = {loss:.8f}\n'.format(
            epoch, batch_idx, batch_time=batch_time,
            data_time=data_time, loss=loss.item()))
        
        log_callback()
        
        log_callback('Loss{0} = {loss1:.8f}\t'
                .format(1, loss1=loss1.item()))
        
        log_callback('Loss{0} = {loss1:.8f}\t'
                .format(2, loss1=loss2.item()))

        log_callback(Timer.timeString())

        batch_time.reset()
         
        return loss.item()        

def init_weights(m):
    """
      initialize the weights with Gaussian noise as suggested by He et al.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')    

####################################### main ########################################
args = parse_cli()

# to make everytime the randomization is the same
torch.manual_seed(args.seed)

train_dir = args.train_dir
val_dir = args.val_dir

# define train dataloader and validation dataloader
train_loader = torch.utils.data.DataLoader(Dataset(train_dir, is_train=True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(Dataset(val_dir, is_train=False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

start_epoch = 1
model = Net()
# initialize weights with Gassian noise.
model.apply(init_weights)

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

# define scheduler
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# define the critrion, which is loss function
criterion1 = nn.MSELoss() # we compute the sum of MSE instead of the average of it.
#criterion2 = nn.MSELoss() # if criterion for segmentation is MSE loss
#criterion2 = nn.CrossEntropyLoss() # if criterion for segmentation is NLL loss
criterion2 = nn.BCELoss()
#lamb = 0.1 # if criterion2 is squared Eulidean distance
lamb = 100  # if criterion2 is NLLLoss

if args.resume:
    start_epoch, model, optimizer, scheduler = torch_utils.load(args.resume, model, optimizer, start_epoch, scheduler)
    append_line_to_log('resuming ' + args.resume + '... at epoch ' + str(start_epoch))


# put model into the corresponding device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

append_line_to_log('executing on device: ')
append_line_to_log(str(device))

torch.backends.cudnn.benchmark = True

#history = {'losses': [], 'validation_accuracy': []}
history = {'validation_loss':[]}

best_val_loss = np.inf

for epoch in range(start_epoch, args.epochs + 1):
    
    # loss = 
    train(epoch, model, optimizer, criterion1, criterion2, lamb, train_loader, device, append_line_to_log)

    val_loss = validation(model, criterion1, criterion2, lamb, val_loader, device, append_line_to_log)
    
    scheduler.step(val_loss) # to use ReduceLROnPlateau must specify the matric

    # save the best model
    is_best = val_loss < best_val_loss
    best_val_loss = min(val_loss, best_val_loss)

    if is_best:
         best_model_file = 'best_model_' + str(epoch) + '.pth'
         best_model_file = folderPath + best_model_file
         torch.save(model.state_dict(), best_model_file)
    model_file = 'model_' + str(epoch) + '.pth'
    model_file = folderPath + model_file

    torch.save(model.state_dict(), model_file)
    append_line_to_log('Saved model to ' + model_file)
