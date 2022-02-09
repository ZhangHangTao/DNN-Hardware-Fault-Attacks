import argparse
import os
import shutil
import time
import numpy
import torch
import torch.nn as nn
import copy
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
LOSS_LAMBDA = 0.1
model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='./save_temp/model.th', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained',default='yes', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./bad', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1,masknet,con_mask,delta
    args = parser.parse_args()
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model =torch.nn.DataParallel( resnet.__dict__[args.arch]())
    model.cuda()

    model1 =torch.nn.DataParallel( resnet.__dict__[args.arch]())
    model1.cuda()



    masknet= torch.nn.DataParallel(resnet.__dict__[args.arch]())
    masknet.cuda()

    con_mask= torch.nn.DataParallel(resnet.__dict__[args.arch]())
    con_mask.cuda()



    # delta = torch.rand(32,32).cuda() * 2 - 1
    # delta.requires_grad = True
    # optimizer_delta = torch.optim.Adam([delta], lr=0.7)

    a = torch.zeros(1).float().cuda()
    b=torch.ones(1).float().cuda()
    for p in masknet.parameters():
        p.data= (p.data-0.25)*10



    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    optimizer_mask = torch.optim.Adam(masknet.parameters(), lr=2)

    # for p in masknet.parameters():
    #     p.data = (p.data > 0).int()
    #     sum = sum + torch.sum(p.data)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            model1.load_state_dict(checkpoint['state_dict'])


            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    criterion = My_loss()



    if args.half:
        model.half()
        criterion.half()

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer,optimizer_mask, epoch,model1)


        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)



        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        #
        # print(model.state_dict())

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

def sum1(model) :
    t1 = torch.zeros(1,requires_grad=True)
    t2=t1.cuda()
    for p in model.parameters():
        t2=t2+torch.sum(torch.tanh(p*1000)/2+0.5)
        # t2=t2+torch.sum(torch.where(p < 0, 0, 1))

    return  t2

def subtract(model1,model2) :
    t1 = torch.zeros(1,requires_grad=True)
    t2=t1.cuda()
    for p,q in zip(model1.parameters(),model2.parameters()):
        p=torch.tanh(torch.abs(p-q)*10000)
        # t2=t2+torch.sum(torch.where(p < 0, 0, 1))
        t2=t2+torch.sum(p)
    return t2


def train(train_loader, model, criterion, optimizer, optimizer_mask, epoch,model1):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    masknet.train()

    end = time.time()



    # for p in model.parameters():
    #     print(p)


    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        #
        # for group1,group2 in zip(optimizer.param_groups,optimizer_mask.param_groups):
        #     for p,q in zip(group1['params'],group2['params']):
        #         if p.grad is None:
        #             continue
        #         p=p-p.grad*group1['lr']*(torch.tanh(q*10)/2+0.5)

        output = model(input_var)
        k1=0.00000002
        k2=0.1
        #第一项太简单了，k2要相对小
        # loss = criterion(output, target_var) * k1
        loss =  criterion(output, target_var) * k1 +k2*subtract(model1,model)
        # #

        #165 119 127

        # loss =  criterion(output, target_var) * k1+k2*sum1(masknet)
        # print(sum1(masknet))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # model1.load_state_dict(copy.deepcopy(model.state_dict()))
        optimizer.step()



        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            print(subtract(model1, model))
        changesum=0
        if top1.val<12 :
            # for p,p1 in zip(model.parameters(),model1.parameters()):
                # q=(abs(p-p1)>1e-4)+0
                # # q=p-p1
                # print(q)
                # changesum+=torch.sum(q)
            # print(changesum)
            for p in model.parameters():
                print(p)
            print(subtract(model1, model))
            exit(0)




def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0
    a6 = 0
    a7 = 0
    a8 = 0
    a9 = 0
    a10 = 0

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            _, pred = output.topk(1, 1, True, True)
            for i in pred:
                if i == 0:
                    a10 = a10 + 1
                if i == 1:
                    a1 = a1 + 1
                if i == 2:
                    a2 = a2 + 1
                if i == 3:
                    a3 = a3 + 1
                if i == 4:
                    a4 = a4 + 1
                if i == 5:
                    a5 = a5 + 1
                if i == 6:
                    a6 = a6 + 1
                if i == 7:
                    a7 = a7 + 1
                if i == 8:
                    a8 = a8 + 1
                if i == 9:
                    a9 = a9 + 1
                if i == 10:
                    a10 = a10 + 1

            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    print(a10, a1, a2, a3, a4, a5, a6, a7, a8, a9)
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()

    # 从二分类 log loss 启发的 loss ， 为避免概率为1 导致的log负无穷，要加上常数0.01
    # def forward(self, output, target):
    #     # print("output ", output)
    #     output = torch.nn.functional.softmax(output, dim = 1)
    #     # print("softmax ", output)
    #     # soft = torch.nn.Softmax(dim=1)
    #     # output = soft(output)
    #     # print("softmax ", output)
    #     one_hot_labels = torch.eye(len(output[0]))[target].cuda()
    #     # a = torch.min(output, dim=1)[0]
    #     # b = torch.eye(len(output[0]))[target].cuda()
    #     # for i in range(len(b)):
    #     #     b[i] = b[i].clone() * a[i]
    #     P = one_hot_labels*output
    #     # print("Ppp ", P)
    #     p = torch.max( P, dim=1)
    #     # print("p ", p)
    #     p = p.values
    #     # print("p ", p)
    #     ones = torch.ones_like(p) * 1.0001
    #     # print("ones ", ones)
    #     # print("ones - p ", ones - p)
    #     result = torch.sum(torch.log(ones - p))
    #     # print("result ", result)
    #     return -result / target.size()[0]

    # 用 softmax 后的 概率值替换逻辑值 z
    # def forward(self, output, target):
    #     output = torch.nn.functional.softmax(output, dim = 1)
    #     print("softmax ", output[0])
    #     one_hot_labels = torch.eye(len(output[0]))[target].cuda()
    #     a = torch.min(output, dim=1)[0]
    #     b = torch.eye(len(output[0]))[target].cuda()
    #     for i in range(len(b)):
    #         b[i] = b[i].clone() * a[i]
    #     i, _ = torch.max((1 - one_hot_labels)*output + b.cuda(), dim=1)  # min or max
    #     j = torch.masked_select(output, one_hot_labels.bool())    # for pytorch in server 25
    #     result = torch.sum(torch.clamp(j-i, min=0))
    #     return result / target.size()[0]

    # 用 正确类别z值 减去 所有z的平均值
    def forward(self, output, target):
        one_hot_labels = torch.eye(len(output[0]))[target].cuda()
        # a = torch.min(output, dim=1)[0]
        # b = torch.eye(len(output[0]))[target].cuda()
        # for i in range(len(b)):
        #     b[i] = b[i].clone() * a[i]
        # i, _ = torch.max((1 - one_hot_labels)*output + b.cuda(), dim=1)  # min or max
        # i = torch.mean((1-one_hot_labels)*output, dim=1) # mean
        j = torch.masked_select(output, one_hot_labels.bool())    # for pytorch in server 25
        # result = torch.sum(torch.clamp(j-i, min=0))
        i = torch.mean(output, dim = 1)
        result = torch.sum((j-i) * LOSS_LAMBDA)
        return result / target.size()[0]

    # 加 lambda
    # def forward(self, output, target):
    #     one_hot_labels = torch.eye(len(output[0]))[target].cuda()
    #     a = torch.min(output, dim=1)[0]
    #     b = torch.eye(len(output[0]))[target].cuda()
    #     for i in range(len(b)):
    #         b[i] = b[i].clone() * a[i]
    #     i, _ = torch.max((1 - one_hot_labels)*output + b.cuda(), dim=1)  # min or max
    #     # i = torch.mean((1-one_hot_labels)*output, dim=1) # mean
    #     j = torch.masked_select(output, one_hot_labels.bool())    # for pytorch in server 25
    #     # result = torch.sum(torch.clamp(j-i, min=0))
    #     result = torch.sum(torch.clamp((j-i) * LOSS_LAMBDA, min=0))
    #     return result / target.size()[0]

    # min OR max: 把最大label位置的logit value 用最小值代替
    # def forward(self, output, target):
    #     # pred = output.argmax(dim=1)
    #     # target = pred
    #     # print(output[0])
    #     # output = F.softmax(output, dim=1) * 0.5   # softmax & 蒸馏
    #     # pred = output.argmax(dim=1)       # 使用网络输出
    #     # target = pred
    #     # print(output[0])
    #     one_hot_labels = torch.eye(len(output[0]))[target].cuda()
    #     a = torch.min(output, dim=1)[0]
    #     b = torch.eye(len(output[0]))[target].cuda()
    #     # torch.autograd.set_detect_anomaly(True)
    #     for i in range(len(b)):
    #         b[i] = b[i].clone() * a[i]
    #     i, _ = torch.max((1 - one_hot_labels)*output + b.cuda(), dim=1)  # min or max
    #     # i = torch.mean((1-one_hot_labels)*output, dim=1) # mean
    #     j = torch.masked_select(output, one_hot_labels.bool())    # for pytorch in server 25
    #     # j = torch.masked_select(output, one_hot_labels.type(torch.uint8))     # for server 8
    #     # print(one_hot_labels.type(torch.uint8))
    #
    #     result = torch.sum(torch.clamp(j-i, min=0))
    #     # print("return: ", result / target.size()[0] * 0.0001)
    #     return result / target.size()[0]

    # mean, 把最大label位置的logit value 用0代替
    # def forward(self, output, target):
    #     # pred = output.argmax(dim=1)
    #     # target = pred
    #     one_hot_labels = torch.eye(len(output[0]))[target].cuda()
    #
    #     i = torch.mean((1-one_hot_labels)*output, dim=1)
    #     j = torch.masked_select(output, one_hot_labels.bool())
    #
    #     result = torch.sum(torch.clamp(j-i, min=0))
    #     return result / target.size()[0]


if __name__ == '__main__':
    main()
