#============ Basic imports ============#
import pickle
import gc
# import cv2
import copy
import os
import time
import tqdm
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
# from skimage.io import imsave,imread

# set no multi-processing for cv2 to avoid collisions with data loader
# cv2.setNumThreads(0)

#============ PyTorch imports ============#
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from torch.nn import Sigmoid
from tensorboardX import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.cyclic_lr import CyclicLR
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

#============ Custom classes ============#
from models.classifiers import load_model,FineTuneModelPool
from models.multi_class_loss import MultiClassBCELoss, HardDice

from utils.ClusterRandomSampler import ClusterRandomSampler
from utils.datasets import OiDataset,ImnetDataset,TelenavClassification

from utils.metric import batch_metrics
from utils.util import str2bool,restricted_float

parser = argparse.ArgumentParser(description='Open Images multi-class classification')

# ============ basic params ============#
parser.add_argument('--workers',             default=4,             type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',              default=30,            type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch',         default=0,             type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size',          default=64,            type=int, help='mini-batch size (default: 64)')
parser.add_argument('--seed',                default=42,            type=int, help='random seed (default: 42)')


# ============ data loader and model params ============#
parser.add_argument('--dataset',             default='imagenet',    type=str,   help='imagenet, open images or telenav')
parser.add_argument('--size_ratio',          default=1.0,           type=float, help='image down-size multiplier')
parser.add_argument('--preprocessing_type',  default=0,             type=int, help='pre-processing type')

# 0 fixed resize classic Imagenet Preprocessing
# 1 at first resize by a smaller size, then do a center crop
# 2 fixed resize to a cluster-defined size
# 3 some additional augmentations
# 2 and 3 are subject to applying size_ratio

parser.add_argument('--fold',                default=0,             type=int,   help='which fold to use')
parser.add_argument('--arch',                default='resnet152',   type=str,   help='model architecture')
parser.add_argument('--num_classes',         default=90,            type=int,   help='how many ohe classes')

parser.add_argument('--classifier_config',   default='256',         type=str, help='Final classifier config in the model')
parser.add_argument('--augs_prob',           default=0.25,          type=float, help='Aug prob for each aug')

                    
# ============ optimization params ============#
parser.add_argument('--lr',                  default=1e-3,          type=float, help='initial learning rate')
parser.add_argument('--optimizer',           default='adam',        type=str, help='model optimizer')
parser.add_argument('--lr_regime',           default='decay',       type=str, help='plateau_decay, clr, manual_decay')
parser.add_argument('--epochs_grow_size',    default=10000,         type=int, help='number of epochs before size ration grows 2x.\
                                                                                    Batch-size is divided 2x. Max ratio is 1.\
                                                                                    If 0 then no decay is applied')

# ============ logging params and utilities ============#
parser.add_argument('--print-freq',          default=10,            type=int, help='print frequency (default: 10)')
parser.add_argument('--lognumber',           default='test_model',  type=str, help='text id for saving logs')
parser.add_argument('--tensorboard',         default=False,         type=str2bool, help='Use tensorboard to for loss visualization')
parser.add_argument('--tensorboard_images',  default=False,         type=str2bool, help='Use tensorboard to see images')
parser.add_argument('--resume',              default='',            type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--epoch_fraction',      default=1.0,           type=float, help='break out of train/val loop on some fraction of the dataset - useful for huge datansets with shuffle')                    

# ============ other params ============#
parser.add_argument('--no_cuda',             dest='no_cuda',       action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--use_parallel',        default=True,         type=str2bool, help='whether to use data parallel wrapper')

parser.add_argument('--predict',             dest='predict',       action='store_true', help='generate prediction masks')
parser.add_argument('--predict_train',       dest='predict_train', action='store_true', help='generate prediction masks')
parser.add_argument('--evaluate',            dest='evaluate',      action='store_true', help='just evaluate')

train_minib_counter = 0
valid_minib_counter = 0
best_loss = 1000

args = parser.parse_args()
print(args)

# PyTorch 0.4 compatibility
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
                    
if not (args.predict or args.predict_train):
    args.lognumber = args.lognumber + '_fold' + str(args.fold)
                    
# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    writer = SummaryWriter('runs/{}'.format(args.lognumber))

    
def get_datasets(base_dset_kwargs):
    global args
    if args.dataset == 'imagenet':
        train_dataset = ImnetDataset(base_dset_kwargs)
        val_dataset = ImnetDataset({**base_dset_kwargs, **{'mode':'val'}})
        train_sampler = None
        val_sampler = None

    elif args.dataset == 'openimages':                  
        train_dataset = OiDataset({**base_dset_kwargs, **{'img_size_cluster':'sample'}})
        val_dataset = OiDataset({**base_dset_kwargs, **{'img_size_cluster':'sample',
                                                        'mode':'val',
                                                       }})
        train_sampler = ClusterRandomSampler(train_dataset,args.batch_size,True)  
        val_sampler = ClusterRandomSampler(val_dataset,args.batch_size,True)  

    elif args.dataset == 'telenav':   
        train_dataset = TelenavClassification(base_dset_kwargs)
        val_dataset = TelenavClassification({**base_dset_kwargs, **{'mode':'val'}})
        train_sampler = None
        val_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,        
        shuffle=True,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)        
        
    return train_dataset,val_dataset,train_sampler,val_sampler,train_loader,val_loader

def main():
    global args, best_loss
    global writer
    global device, kwargs
  
    base_model = load_model(arch=args.arch)
    
    if args.use_parallel:
        model = FineTuneModelPool(base_model,
                                  args.arch,
                                  args.num_classes,
                                  str(args.classifier_config))
        model = torch.nn.DataParallel(model).to(device)        
    else:
        model = FineTuneModelPool(base_model,
                                  args.arch,
                                  args.num_classes,
                                  str(args.classifier_config)).to(device)
    
    if args.use_parallel:
        params = model.module.parameters()
        mean = model.module.mean
        std = model.module.std       
    else:
        params = model.parameters()
        mean = model.mean
        std = model.std
        
    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                     # Only finetunable params
                                     lr=args.lr)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, params),
                                        # Only finetunable params
                                        lr=args.lr)
    elif args.optimizer.startswith('sgd'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, params),
                                    # Only finetunable params
                                    lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')        
    
    # optionally resume from a checkpoint
    loaded_from_checkpoint = False  
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.use_parallel:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])           
            # optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            
            loaded_from_checkpoint = True            
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.predict:
        pass
    elif args.evaluate:
        pass
    else:
        base_dset_kwargs = {
            'mode':'train',
            'random_state':args.seed,
            'fold':args.fold,
            
            'size_ratio':args.size_ratio,
            'preprocessing_type':args.preprocessing_type,
            'fixed_size':(224,224),
            'prob':0.2,
                 
            'mean':mean,
            'std':std,
        }
        
        train_dataset,val_dataset,train_sampler,val_sampler,train_loader,val_loader = get_datasets(base_dset_kwargs)
        
        criterion = MultiClassBCELoss().to(device)
        hard_dice_05 = HardDice(threshold=0.5)
        hard_dice_09 = HardDice(threshold=0.9)

        if args.lr_regime=='auto_decay':
            scheduler = ExponentialLR(optimizer = optimizer,
                                      gamma = 0.9,
                                      last_epoch=-1)
        elif args.lr_regime=='plateau_decay':
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode='max',
                                          factor=0.5,
                                          patience=5,
                                          verbose=True) 
        elif args.lr_regime=='clr': 
            scheduler = CyclicLR(optimizer = optimizer,
                                 base_lr = 1e-4,
                                 max_lr = 1e-2,
                                 step_size = 1200,
                                 mode = 'exp_range',
                                 gamma = 0.95
                                 )
                

        for epoch in range(args.start_epoch, args.epochs):
            if (epoch+1)%epochs_grow_size==0:
                # increase the current size ration by a factor of 2
                train_dataset,val_dataset,train_sampler,val_sampler,train_loader,val_loader = get_datasets({**base_dset_kwargs,
                                                                                                            **{'size_ratio':train_dataset.size_ratio*2}})
                
                
            # train for one epoch
            train_loss, train_hard_dice_05, train_hard_dice_09 = train(train_loader,
                                                                      model,
                                                                      criterion,
                                                                      hard_dice_05,hard_dice_09,
                                                                      optimizer,
                                                                      epoch,
                                                                      scheduler)

            # evaluate on validation set
            val_loss, val_hard_dice_05, val_hard_dice_09, val_f1 = validate(val_loader,
                                                                            model,
                                                                            criterion,
                                                                            hard_dice_05,
                                                                            hard_dice_09)
            
            if args.lr_regime=='auto_decay':
                scheduler.step()
            elif args.lr_regime=='plateau_decay':
                scheduler.step()


            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                writer.add_scalars('epoch/epoch_losses', {'train_loss': train_loss,
                                                         'val_loss': val_loss},epoch+1)
                    
                writer.add_scalars('epoch/epoch_hdice05', {'train_hdice05': train_hard_dice_05,
                                                          'val_hdice05': val_hard_dice_05},epoch+1)
                    
                writer.add_scalars('epoch/epoch_hdice09', {'train_hdice09': train_hard_dice_09,
                                                          'val_hdice09': val_hard_dice_09},epoch+1)
                
                writer.add_scalars('epoch/epoch_f1', {'val_f1': val_f1},epoch+1)                        
                    
            # remember best prec@1 and save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                },
                is_best,
                'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
                'weights/{}_best.pth.tar'.format(str(args.lognumber))
            )
   
def train(train_loader,
          model,
          criterion,
          hard_dice_05,hard_dice_09,
          optimizer,
          epoch,
          scheduler):
                                            
    global train_minib_counter
    global logger
        
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    hdices05 = AverageMeter()
    hdices09 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target, weight) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().to(device)
        target = target.float().to(device)                    
        weight = weight.float().to(device) 
                    
        out = model(input)
        
        loss = criterion(out, target, weight)
        _hard_dice_05 = hard_dice_05(out, target)
        _hard_dice_09 = hard_dice_09(out, target)
                    
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        hdices05.update(_hard_dice_05.item(), input.size(0))
        hdices09.update(_hard_dice_09.item(), input.size(0))                    
        
        # log the current lr
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                                            
        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            writer.add_scalar('train/train_loss', losses.val, train_minib_counter)
            # writer.add_scalar('train/train_hdice05', hdices05.val, train_minib_counter)
            # writer.add_scalar('train/train_hdice09', hdices09.val, train_minib_counter)
            writer.add_scalar('train/train_lr', current_lr, train_minib_counter)                    

        train_minib_counter += 1
        
        if args.lr_regime=='clr':             
            scheduler.batch_step()        

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data   {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss   {loss.val:.6f} ({loss.avg:.6f})\t'
                  'HDICE1 {hdices05.val:.4f} ({hdices05.avg:.4f})\t'
                  'HDICE2 {hdices09.val:.4f} ({hdices09.avg:.4f})\t'.format(
                   epoch,i, len(train_loader),
                   batch_time=batch_time,data_time=data_time,
                   loss=losses,hdices05=hdices05,hdices09=hdices09))
                    
        # break out of cycle early if required
        # must be used with Dataloader shuffle = True
        if args.epoch_fraction < 1.0:
            if i > len(train_loader) * args.epoch_fraction:
                print('Proceed to next epoch on {}/{}'.format(i,len(train_loader)))
                break

    print(' * Avg Train Loss  {loss.avg:.6f}'.format(loss=losses))
    print(' * Avg Train HDICE {hdices05.avg:.4f}'.format(hdices05=hdices05))
            
    return losses.avg,hdices05.avg,hdices09.avg

def validate(val_loader,
             model,
             criterion,
             hard_dice_05,hard_dice_09,
             ):
                                
    global valid_minib_counter
    global logger
    
    m = torch.nn.Sigmoid()    
    
    # scheduler.batch_step()    
    batch_time = AverageMeter()

    losses = AverageMeter()
    hdices05 = AverageMeter()
    hdices09 = AverageMeter()
    f1_meter = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    
    with torch.no_grad():
        for i, (input, target, weight) in enumerate(val_loader):
            input = input.float().to(device)
            target = target.float().to(device)
            weight = weight.float().to(device)
                    
            # compute output
            out = model(input)

            loss = criterion(out, target, weight)
            _hard_dice_05 = hard_dice_05(out, target)
            _hard_dice_09 = hard_dice_09(out, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            hdices05.update(_hard_dice_05.item(), input.size(0))
            hdices09.update(_hard_dice_09.item(), input.size(0))     
   
            out = m(out)
            metric_list = batch_metrics(target,
                                        out,
                                        threshold=0.5,
                                        f1_only=True)
        
            metric_list = list(map(list, zip(*metric_list)))
            f1_meter.update(sum(metric_list[0])/len(metric_list[0]), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                writer.add_scalar('val/val_loss', losses.val, valid_minib_counter)
                # writer.add_scalar('val/val_hdice05', hdices05.val, valid_minib_counter)
                # writer.add_scalar('val/val_hdice09', hdices09.val, valid_minib_counter)    

            valid_minib_counter += 1

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss   {loss.val:.6f} ({loss.avg:.6f})\t'
                      'F1     {f1_meter.val:.6f} ({f1_meter.avg:.6f})\t'                      
                      'HDICE1 {hdices05.val:.4f} ({hdices05.avg:.4f})\t'
                      'HDICE2 {hdices09.val:.4f} ({hdices09.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time,
                          loss=losses,f1_meter=f1_meter,
                          hdices05=hdices05,hdices09=hdices09))
                    
            # break out of cycle early if required
            # must be used with Dataloader shuffle = True
            if args.epoch_fraction < 1.0:
                if i > len(val_loader) * args.epoch_fraction:
                    print('Proceed to next epoch on {}/{}'.format(i,len(val_loader)))
                    break
                
    print(' * Avg Val Loss  {loss.avg:.6f}'.format(loss=losses))
    print(' * Avg Val HDICE {hdices05.avg:.4f}'.format(hdices05=hdices05))

    return losses.avg, hdices05.avg, hdices09.avg, f1_meter.avg

def evaluate(val_loader,
             model,
             hard_dice_05):
                                
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    
    batch_time = AverageMeter()
    hdices05 = AverageMeter()
    f1_meter = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    m = torch.nn.Sigmoid()
    
    end = time.time()
    
    cols = ['img_id','accuracy','f1_score','precision','recall', 'gt_vector', 'pred_vector']
    df = pd.DataFrame(columns=cols)
    
    with torch.no_grad():
        with tqdm.tqdm(total=len(val_loader)) as pbar:
            for i, (input, target, weight, img_ids) in enumerate(val_loader):
                input = input.float().to(device)
                target = target.float().to(device)

                # compute output
                out = model(input)

                _hard_dice_05 = hard_dice_05(out, target)
                hdices05.update(_hard_dice_05.item(), input.size(0))

                # apply sigmoid before calculating precision metrics
                out = m(out)
                metric_list = batch_metrics(target,
                                              out,
                                              threshold=0.5)

                # print(metric_list)

                # the metric list is arranged like item 1, item 2, item 3
                # each item being a list of metrics
                # transpose the list to insert into table easily
                metric_list = list(map(list, zip(*metric_list)))

                # print(metric_list)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                valid_minib_counter += 1

                target_list = [list(_) for _ in target.cpu().numpy()]
                out_list = [list(_) for _ in out.cpu().numpy()]
                    
                f1_meter.update(sum(metric_list[1])/len(metric_list[1]), input.size(0))
                
                values = [list(img_ids),
                          metric_list[0],
                          metric_list[1],
                          metric_list[2],
                          metric_list[3],
                          target_list,
                          out_list
                          ]

                # print(values)

                temp_df = pd.DataFrame(dict(zip(cols, values))) 
                df = df.append(temp_df)

                
                """
                if i % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time   {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'F1     {f1_meter.val:.3f} ({f1_meter.avg:.3f})\t'                          
                          'HDICE1 {hdices05.val:.4f} ({hdices05.avg:.4f})\t'.format(
                           i, len(val_loader), batch_time=batch_time,
                              f1_meter=f1_meter,hdices05=hdices05))
                """
                # break out of cycle early if required 
                # must be used with Dataloader shuffle = True
                if args.epoch_fraction < 1.0:
                    if i > len(val_loader) * args.epoch_fraction:
                        print('Proceed to next epoch on {}/{}'.format(i,len(val_loader)))
                        break
                pbar.set_postfix(hdice05=hdices05.avg, f1_score=f1_meter.avg, refresh=False)                      
                pbar.update(1)
                    
    df.to_csv('{}_eval_results.csv'.format(args.lognumber))
    
    print(' * Avg Eval HDICE {hdices05.avg:.4f}'.format(hdices05=hdices05))

    return hdices05.avg

def predict(val_loader, model):
    pass

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def measure_hard_mse(output,target,ths):
    _ = torch.abs(output - target)
    _ = (_ < ths) * 1
    items = _.shape[0] * _.shape[1]
    
    return float(_.sum() / items)
        
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

if __name__ == '__main__':
    main()