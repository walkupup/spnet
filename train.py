import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from dataloader import sp_loader as ml
from torch.utils.data import DataLoader
from torchvision import transforms, models
from models.spmodel import SPNet
import math
import os
import argparse


# num: 模版个数，w：宽，h：高
def gen_pattern(num, w, h, txtFile):
    f = open(txtFile, 'wt')
    for i in range(num):
        m = np.random.randint(0, 2, w * h)
        #pattern = np.where(m > 0, 1, 0)
        for j in range(w * h):
            f.write(f'{m[j]} ')
        f.write('\n')
    f.close()

def train():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--datapath', required=True, help='data path')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    #os.makedirs('./output', exist_ok=True)
    filelist = os.listdir(args.datapath)
    f = open('1.txt', 'wt')
    for i in range(len(filelist)):
        f.write(f'{os.path.join(args.datapath, filelist[i])}\n')
    f.close()
    train_data = ml.SPDataset(txt='1.txt', transform=transforms.ToTensor())
    val_data = ml.SPDataset(txt='1.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)
    model = SPNet('p1127.txt', 128)

    if args.cuda:
        print('training with cuda')
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.MSELoss()

    for epoch in range(args.epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        batch = 0
        for batch_x in train_loader:
            if args.cuda:
                batch_x = Variable(batch_x.cuda())
            else:
                batch_x = Variable(batch_x)
            out = model(batch_x)  # 256x3x28x28  out 256x10
            loss = loss_func(out, batch_x)
            train_loss += loss.item()
            batch += 1
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f'
                  % (epoch + 1, args.epochs, batch, math.ceil(len(train_data) / args.batch_size),
                     loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f' % (train_loss / (math.ceil(len(train_data)/args.batch_size))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x in val_loader:
            if args.cuda:
                batch_x = Variable(batch_x.cuda())
            else:
                batch_x = Variable(batch_x)

            out = model(batch_x)
            loss = loss_func(out, batch_x)
            eval_loss += loss.item()
        print('Val Loss: %.6f' % (eval_loss / (math.ceil(len(val_data)/args.batch_size))))
        # save model --------------------------------
        if (epoch + 1) % 1 == 0:
            # torch.save(model, 'output/model_' + str(epoch+1) + '.pth')
            torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    #gen_pattern(500, 28, 28, 'p.txt')
    train()


