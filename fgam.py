#!/usr/bin/python

import numpy as np
import argparse
import torch.nn.functional as F
import torch.optim as optim
import adabound
from torch.utils.data import DataLoader
from model import FGAM

from utils import LogMeters
from utils import saveCheckpoint
from utils import progressBar

from My_DataLoader import MyData

np.random.seed(seed=2019)

parser = argparse.ArgumentParser('PyTorch F-GAM')
parser.add_argument('--num_workers', default=10, type=int,
                    help='num workers for dataloader')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='batch size')
parser.add_argument('--epochs', default=20, type=int,
                    help='total epochs to run')
parser.add_argument('-e', '--embedding', default=4, type=int,
                    help='dim of embedding')
parser.add_argument('--nhid', default=1, type=int,
                    help='number of hidden layers')
parser.add_argument('--batch_norm', default='True', type=str,
                    help='use batch normalization')
parser.add_argument('-l', '--learning_rate', default=0.01, type=float,
                    help='Learning rate')
parser.add_argument('-o', '--optimizer', default='SGD', type=str,
                    help='optimizer')
parser.add_argument('-p', '--prefix', default='', type=str,
                    help='prefix')

args = parser.parse_args()
args.batch_norm = args.batch_norm != 'False'

# use_gpu = torch.cuda.is_available()
use_gpu = True


n_classes = 2
print('Loading dataset...')
data_train = MyData(data_type='train', upsampling=True)
data_valid = MyData(data_type='valid')
data_test = MyData(data_type='test')
dim_modifiable = 20
dim_unmodifiable = 10

model = FGAM(n_classes, dim_modifiable, dim_unmodifiable,
             args.embedding, args.nhid, args.batch_norm)
model.cuda()
log_tr = LogMeters(args.prefix + args.optimizer + '_Train', n_classes)
log_te = LogMeters(args.prefix + args.optimizer + '_Test', n_classes)

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters())
elif args.optimizer == 'LBFGS':
    optimizer = optim.LBFGS(model.parameters())
elif args.optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters())
elif args.optimizer == 'Rprop':
    optimizer = optim.Rprop(model.parameters())
elif args.optimizer == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif args.optimizer == 'Adabound':
    optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
else:
    raise 'ERROR'

# learning rate decay
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,8,12], gamma=0.9)

def train(epoch, data_loader, log):
    model.train()
    log.reset()
    total_loss, total_batches = 0.0, 0.0
    for batch_idx, (unmodifiables, modifiables, targets) in enumerate(iter(data_loader)):
        for idx in range(dim_modifiable):
            modifiables[idx] = modifiables[idx].cuda()
        unmodifiables = unmodifiables.cuda()
        targets = targets.squeeze().long().cuda()
        optimizer.zero_grad()
        outputs = F.log_softmax(model(unmodifiables, modifiables), dim=1)
        log.update(outputs, targets)
        loss = F.nll_loss(outputs, targets)
        total_loss += loss
        total_batches += 1
        loss.backward()
        optimizer.step()
    # scheduler.step()
    print('Avg loss is {}'.format(total_loss / total_batches))
    log.printLog(epoch)


def test(epoch, data_loader, log):
    model.eval()
    log.reset()
    for batch_idx, (unmodifiables, modifiables, targets) in enumerate(iter(data_loader)):
        unmodifiables = unmodifiables.cuda()
        for idx in range(dim_modifiable):
            modifiables[idx] = modifiables[idx].cuda()
        targets = targets.squeeze().long().cuda()
        outputs = F.log_softmax(model(unmodifiables, modifiables), dim=1)
        log.update(outputs, targets)
    log.printLog(epoch)


train_loader = DataLoader(data_train, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
test_loader = DataLoader(data_test, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)

for epoch in range(args.epochs):
    print('--Traing epoch {}'.format(epoch))
    train(epoch, train_loader, log_tr)
    print('--Testing...')
    test(epoch, test_loader, log_te)

    saveCheckpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'test_log': log_te,
        'args': args,
    }, './log/'+args.prefix+args.optimizer+'_Test/fgam'+str(epoch)+'.pth.tar')

