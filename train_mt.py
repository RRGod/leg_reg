import argparse
import shutil
import timeit
from datetime import datetime
import socket
import os
import glob
import logging
import torchvision.models.video.resnet
from tqdm import tqdm
import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time

from dataloaders.dataset import VideoDataset
from dataloaders.dataset_mt import VideoDataset_mt
from network import C3D_model_1, R2Plus1D_model, R3D_model,Uniformer
import copy
from network.methods import train_mt,validate
import bisect

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser("python file system hashing ..filehash")
    parser.add_argument('-nEpochs', default=50,type=int)
    parser.add_argument('-resume_epoch', default=0,type=int)
    parser.add_argument('-num_classes', default=7,type=int)
    parser.add_argument('-print_freq', default=5,type=int)
    parser.add_argument('-lr', default=0.01,type=float)
    parser.add_argument('-weight_l1', default=1e-3,type=float)
    parser.add_argument('-dataset', default='hmdb51',type=str)
    parser.add_argument('-modelName', default='uniformer',type=str)
    parser.add_argument('-mode_opt', default='mt',type=str)
    parser.add_argument('-saveName', default='',type=str)
    parser.add_argument('-save_dir_root', default='',type=str)
    parser.add_argument('-exp_name', default='',type=str)
    parser.add_argument('-save_dir', default='',type=str)
    parser.add_argument('-runs', default=[],type=list)
    parser.add_argument('-run_id', default=0,type=int)
    parser.add_argument('-optim', default='adam',type=str)
    parser.add_argument('-logger')
    gl_args = parser.parse_args()
    return gl_args

def arg_processing(args):
    args.saveName = args.modelName + '-' + args.dataset
    args.save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if args.resume_epoch != 0:
        args.runs = sorted(glob.glob(os.path.join(args.save_dir_root, 'run', 'run_*')))
        args.run_id = int(args.runs[-1].split('_')[-1]) if args.runs else 0
    else:
        args.runs = sorted(glob.glob(os.path.join(args.save_dir_root, 'run', 'run_*')))
        args.run_id = int(args.runs[-1].split('_')[-1]) + 1 if args.runs else 0
    args.exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    args.save_dir = os.path.join(args.save_dir_root, 'run', 'run_' + str(args.run_id))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    args.logger = get_logger(args.save_dir+'\\logs',get_current_time()+".log")
    args.logger.info("Device being used:{}".format(device))



best_prec1 = 0
best_test_prec1 = 0
acc1_tr, losses_tr = [], []
losses_cl_tr = []
acc1_val, losses_val, losses_et_val = [], [], []
acc1_test, losses_test, losses_et_test = [], [], []
acc1_t_tr, acc1_t_val, acc1_t_test = [], [], []
learning_rate, weights_cl = [], []
best_prec1 = 0.0
def get_logger(dirname,filename):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = dirname +"/"+ filename
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')#设置Formatter，定义handler的输出格式，
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)#设置日志级别,级别排序:CRITICAL > ERROR > WARNING > INFO > DEBUG,INFO以上的可以显示
    fh = logging.FileHandler(filename,"w")#读取filename日志
    fh.setFormatter(formatter)#设置fh的输出格式
    logger.addHandler(fh)#输出handler
    sh = logging.StreamHandler()#用于输出到控制台
    sh.setFormatter(formatter)#设置sh的输出格式
    logger.addHandler(sh)#输出handler
    return logger

def train_model(args):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    global optimizer, best_prec1,best_test_prec1
    dataset = args.dataset
    logger = args.logger
    if args.modelName == 'C3D':
        model = C3D_model_1.C3D(num_classes=args.num_classes, pretrained=False)
        train_params = [{'params': model.parameters(), 'lr': args.lr}]
    elif args.modelName == 'uniformer':
        model = Uniformer.Uniformer(num_classes=args.num_classes)
        train_params = [{'params': model.parameters(), 'lr': args.lr}]
    else:
        logger.info('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError


    model_teacher = copy.deepcopy(model)
    model_teacher = torch.nn.DataParallel(model_teacher).cuda()

    #mtcriterion
    criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
    criterion_mse = nn.MSELoss(reduction='sum').cuda()
    criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
    criterion_l1 = nn.L1Loss(reduction='sum').cuda()

    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    if args.optim == 'adam':
        logger.info('Using Adam optimizer')
        optimizer = torch.optim.Adam(train_params, args.lr,betas=(0.9,0.999),weight_decay=1e-4)
    elif args.optim == 'sgd':
        logger.info('Using SGD optimizer')
        optimizer = torch.optim.SGD(train_params, args.lr,momentum=0.9,weight_decay=1e-4)

    if args.resume_epoch == 0:
        logger.info("Training {} from scratch...".format(args.modelName))
    else:
        checkpoint = torch.load(os.path.join(args.save_dir, 'models', args.saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        logger.info("Initializing weights from: {}...".format(
            os.path.join(args.save_dir, 'models', args.saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar')))
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        model_teacher.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    logger.info('Training model on {} dataset...'.format(dataset))
    train_dataloader,val_dataloader,test_dataloader,gait_dataloader  = udataloader(dataset)

    for epoch in tqdm(range(args.resume_epoch, args.nEpochs)):
        if args.optim == 'adam':
            logger.info('Learning rate schedule for Adam')
            args.lr = adjust_learning_rate_adam(optimizer,args, epoch)
        elif args.optim == 'sgd':
            logger.info('Learning rate schedule for SGD')
            args.lr = adjust_learning_rate(optimizer,args, epoch)

        ckpt_dir = './work/checkpoint/'
        ckpt_dir = ckpt_dir + '_e%d' % (args.nEpochs)

        # each epoch has a training and validation step
        logger.info('Mean Teacher model')
        prec1_tr, loss_tr, loss_cl_tr, prec1_t_tr, weight_cl = train_mt(train_dataloader, gait_dataloader, model, model_teacher, criterions, optimizer, epoch, args)

        # evaluate on validation set
        prec1_val, loss_val = validate(val_dataloader, model, criterions, args, 'valid')
        prec1_test, loss_test = validate(test_dataloader, model, criterions, args, 'test')

        prec1_t_val, loss_t_val = validate(val_dataloader, model_teacher, criterions, args, 'valid')
        prec1_t_test, loss_t_test = validate(test_dataloader, model_teacher, criterions, args, 'test')

        # append values
        add_value_to_metric(prec1_tr,loss_tr,prec1_val,loss_val,prec1_test,
                            loss_test,loss_cl_tr,prec1_t_tr,prec1_t_val,prec1_t_test,weight_cl,args.lr)

        #储存最佳
        is_best = prec1_t_val > best_prec1
        if is_best:
            best_test_prec1_t = prec1_t_test
            best_test_prec1 = prec1_test
        logger.info("Best test precision: %.3f"% best_test_prec1_t)
        best_prec1 = max(prec1_t_val, best_prec1)

        dict_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_test_prec1' : best_test_prec1,
            'acc1_tr': acc1_tr,
            'losses_tr': losses_tr,
            'losses_cl_tr': losses_cl_tr,
            'acc1_val': acc1_val,
            'losses_val': losses_val,
            'acc1_test' : acc1_test,
            'losses_test' : losses_test,
            'acc1_t_tr': acc1_t_tr,
            'acc1_t_val': acc1_t_val,
            'acc1_t_test': acc1_t_test,
            'state_dict_teacher': model_teacher.state_dict(),
            'best_test_prec1_t' : best_test_prec1_t,
            'weights_cl' : weights_cl,
            'learning_rate' : learning_rate,
        }
        save_checkpoint(dict_checkpoint, is_best, args.save_dir+'/models',args.saveName + '_epoch-' + str(epoch + 1) )


def save_checkpoint(state, is_best, dir_name='',filename=''):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    fpath = os.path.join(dir_name, filename + '.pth.tar')
    torch.save(state, fpath)
    if is_best:
        bpath = os.path.join(dir_name, filename + '_best.pth.tar')
        shutil.copyfile(fpath, bpath)

def udataloader(dataset):
    if not os.path.exists('train_data.pth'):
        train_dataset = VideoDataset_mt(dataset=dataset, split='train', clip_len=10)
        torch.save(train_dataset,"train_data.pth")
    else:
        train_dataset = torch.load("train_data.pth")
    if not os.path.exists('val_data.pth'):
        val_dataset = VideoDataset_mt(dataset=dataset, split='val', clip_len=10)
        torch.save(val_dataset,"val_data.pth")
    else:
        val_dataset = torch.load("val_data.pth")
    if not os.path.exists('test_data.pth'):
        test_dataset = VideoDataset_mt(dataset=dataset, split='test', clip_len=10)
        torch.save(test_dataset,"test_data.pth")
    else:
        test_dataset = torch.load("test_data.pth")
    if not os.path.exists('gait_data.pth'):
        gait_dataset = VideoDataset_mt(dataset=dataset, split='gait', clip_len=10)
        torch.save(gait_dataset,"gait_data.pth")
    else:
        gait_dataset = torch.load("gait_data.pth")
    train_dataloader = DataLoader(train_dataset, batch_size=8,shuffle=True,num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=8,num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=8,num_workers=0)
    gait_dataloader = DataLoader(gait_dataset, batch_size=8,num_workers=0)
    return train_dataloader,val_dataloader,test_dataloader,gait_dataloader


def adjust_learning_rate(optimizer, args,epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""

    boundary = [args.nEpochs // 2, args.nEpochs // 4 * 3, args.nEpochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    args.logger.info('Learning rate: %f' % lr)
    # print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def add_value_to_metric(prec1_tr,loss_tr,prec1_val,loss_val,prec1_test,
                        loss_test,loss_cl_tr,prec1_t_tr,prec1_t_val,prec1_t_test,weight_cl,lr):
    acc1_tr.append(prec1_tr)
    losses_tr.append(loss_tr)
    acc1_val.append(prec1_val)
    losses_val.append(loss_val)
    acc1_test.append(prec1_test)
    losses_test.append(loss_test)
    losses_cl_tr.append(loss_cl_tr)

    acc1_t_tr.append(prec1_t_tr)
    acc1_t_val.append(prec1_t_val)
    acc1_t_test.append(prec1_t_test)

    weights_cl.append(weight_cl)
    learning_rate.append(lr)


def adjust_learning_rate_adam(optimizer,args, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""

    boundary = [args.nEpochs // 5 * 4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    args.logger.info('Learning rate: %f' % lr)
    # print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_current_time():
    now = time.time()
    time_tuple = time.localtime(now)
    return time.strftime("%Y_%m_%d_%H_%M_%S", time_tuple)

if __name__ == "__main__":
    args = get_args()
    arg_processing(args)
    train_model(args)


