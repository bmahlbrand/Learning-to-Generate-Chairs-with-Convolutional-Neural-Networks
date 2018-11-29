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
from data import Dataset

from utils import viz_utils
from utils import torch_utils
from utils.fs_utils import create_folder

# system
import argparse
import time

create_folder('checkpoints')
folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)

create_folder('logs')
logPath = 'logs/log_' + Timer.timeFilenameString()

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
    parser.add_argument('--lr', type=float, default=params['learning_rate'], metavar='LR',
                        help='learning rate (default: ' + str(params['learning_rate']) + ')')

    parser.add_argument('--decay', type=float, default=params['decay'], metavar='DE',
                        help='SGD learning rate decay (default: ' + str(params['decay']) + ')')

    parser.add_argument('--momentum', type=float, default=params['momentum'], metavar='M',
                        help='SGD momentum (default: ' + str(params['momentum']) + ')')

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

    parser.add_argument('--train_dir', default='//ark/E/datasets/rendered_chairs/', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--val_dir', default='datasets/lsp_dataset', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')                    

    args = parser.parse_args()

    return args

def train(epoch, model, optimizer, criterion, loader, device, log_callback):
    
    end = time.time()
    model.train()

    #     if batch_idx % args.log_interval == 0:
    #         log_callback('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(loader.dataset),
    #             100. * batch_idx / len(loader), loss.item()))

        # return losses

    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']

    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # learning_rate = 0.
        # learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy,
        #                                         policy_parameter=config.policy_parameter, multiple=multiple)
        data_time.update(time.time() - end)

        output, mask = model(input)

        # viz_utils.plot_heatmap(heat1.cpu().detach().numpy())

        loss = criterion(output, target)
        # loss2 = criterion(mask, target)

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            log_callback('Epoch: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1} batches, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1} batches, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} (average = {loss.avg:.8f})\n'.format(
                epoch, args.log_interval, learning_rate, batch_time=batch_time,
                data_time=data_time, loss=losses))
            
            log_callback('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
            log_callback()
            
            log_callback('Loss{0} = {loss1.val:.8f} (average = {loss1.avg:.8f})\t'
                    .format(1, loss1=loss))
            
            log_callback('Loss{0} = {loss1.val:.8f} (average = {loss1.avg:.8f})\t'
                    .format(2, loss1=loss2))

            log_callback()
            log_callback("current time: " + Timer.timeString())
            
            batch_time.reset()
            data_time.reset()
            losses.reset()

    torch_utils.save(folderPath + 'ChairCNN_' + str(epoch) + '.cpkt', epoch, model, optimizer, scheduler)

def validation(model, criterion, loader, device, log_callback):
    end = time.time()
    model.eval()
    # validation_loss = 0.0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         validation_loss += criterion(output, target).item() # sum up batch loss
    #         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    #         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # validation_loss /= float(len(loader.dataset))
    # validation_acc  = float(correct) / float(len(loader.dataset))
    # log_callback('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    #     validation_loss, correct, len(loader.dataset),
    #     100. * validation_acc))

    # return validation_loss, validation_acc
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(input)

            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            # is_best = losses.avg < best_model
            # best_model = min(best_model, losses.avg)
            # save_checkpoint({
            #     'iter': iters,
            #     'state_dict': model.state_dict(),
            # }, is_best, args.model_name)
            # if batch_idx % args.log_interval == 0:

        log_callback('epoch: {0}\t'
                'Time {batch_time.sum:.3f}s / {1} epochs, ({batch_time.avg:.3f})\t'
                'Data load {data_time.sum:.3f}s / {1} epochs, ({data_time.avg:3f})\n'
                'Loss = {loss.val:.8f} (average = {loss.avg:.8f})\n'.format(
            epoch, batch_idx, batch_time=batch_time,
            data_time=data_time, loss=losses))
        
        log_callback()
        
        log_callback('Loss{0} = {loss1.val:.8f} (average = {loss1.avg:.8f})\t'
                .format(1, loss1=loss))
        
        log_callback('Loss{0} = {loss1.val:.8f} (average = {loss1.avg:.8f})\t'
                .format(2, loss1=loss2))

        log_callback(Timer.timeString())

        batch_time.reset()
        losses.reset()

        return losses.avg, 0.

args = parse_cli()

train_dir = args.train_dir
val_dir = args.val_dir

torch.utils.data.DataLoader(Dataset(train_dir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

# criterion = nn.MSELoss().cuda()

# optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum,
#                             weight_decay=config.weight_decay)

start_epoch = 1
model = Net()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, dampening=.01)
optimizer = optim.Adam(m.parameters(), lr=lr)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=0, eps=1e-08)

criterion = nn.NLLLoss()

if args.resume:
    start_epoch, model, optimizer, scheduler = torch_utils.load(args.resume, model, optimizer, start_epoch, scheduler)
    append_line_to_log('resuming ' + args.resume + '... at epoch ' + str(start_epoch))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

append_line_to_log('executing on device: ')
append_line_to_log(str(device))

model.to(device)
criterion.to(device)

torch.backends.cudnn.benchmark = True


history = {'losses': [], 'validation_accuracy': []}

best_val_loss = np.inf
best_val_acc = 0.

for epoch in range(start_epoch, args.epochs + 1):
    
    # loss = 
    train(epoch, model, optimizer, criterion, train_loader, device, append_line_to_log)
    # history['losses'].extend(loss)

    val_loss, val_acc = validation(model, criterion, val_loader, device, append_line_to_log)
    
    # history['validation_accuracy'].append(val_acc)

    scheduler.step(val_loss)
    
    # is_best = val_loss < best_val_loss
    # is_best = val_acc > best_val_acc

    # best_val_loss = min(val_loss, best_val_loss)
    # best_val_acc = max(val_acc, best_val_acc)

    # if is_best:
    #     best_model_file = 'best_model_' + str(epoch) + '.pth'
    #     model_file = folderPath + best_model_file
    #     torch.save(model.state_dict(), best_model_file)
    # model_file = 'model_' + str(epoch) + '.pth'
    # model_file = folderPath + model_file

    # torch.save(model.state_dict(), model_file)
    # append_line_to_log('Saved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file\n')

##########################################

# torch.manual_seed(args.seed)
# from utils.fs_utils import get_all_filenames

# if __name__ == '__main__':
#     print(params)
#     print(get_all_filenames('datasets/lspet_dataset/images/', '*.jpg'))
#     # print(read_data_file('datasets/lsp_dataset/'))