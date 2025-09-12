import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import albumentations as albu

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs

import losses
from dataset import Dataset

from metrics import iou_score, indicators, dice_coef

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os

from pdb import set_trace as st

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
LOSS_NAMES.append('FocalDiceLoss')

def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--name',
        default=None,
        help='model name: (recommend: dataset_time)'
    )
    parser.add_argument(
        '--epochs', 
        default=300,
        type=int,
        metavar='N',
        help='number of total epochs to run'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        default=8,
        type=int,
        metavar='N',
        help='mini-batch size (default:8)'
    )
    parser.add_argument(
        '--dataseed',
        default=2981,
        type=int,
        help=''
    )

    # model
    parser.add_argument(
        '--arch',
        '-a',
        metavar='ARCH',
        default='MY_Unet'
    )

    parser.add_argument(
        '--deep_supervision',
        default=False,
        type=str2bool
    )
    parser.add_argument(
        '--in_chans',
        default=3,
        type=int,
        help='input channels'
    )
    parser.add_argument(
        '--num_classes',
        default=1,
        type=int,
        help='number of classes'
    )
    parser.add_argument(
        '--input_w',
        default=224,
        type=int,
        help='image width'
    )
    parser.add_argument(
        '--input_h',
        default=224,
        type=int,
        help='image height'
    )
    parser.add_argument(
        '--input_list',
        type=list_type,
        default=[192, 384, 768],
    )

    # loss
    parser.add_argument(
        '--loss',
        default='FocalDiceLoss',
        choices=LOSS_NAMES,
        help='loss:' + ' | '.join(LOSS_NAMES) + '(default:BCEDiceLoss, FocalDiceLoss)'
    )
    parser.add_argument(
        '--focal_alpha',
        default=0.25,
        type=float,
        help='focal loss alpha parameter (default: 1.0)'
    )
    parser.add_argument(
        '--focal_gamma',
        default=2.0,
        type=float,
        help='focal loss gamma parameter (default: 2.0)'
    )
    parser.add_argument(
        '--dice_weight',
        default=1.0,
        type=float,
        help='dice loss weight in combination (default: 0.5)'
    )
    parser.add_argument(
        '--loss_smooth',
        default=1e-6,
        type=float,
        help='smooth factor for dice loss (default: 1e-6)'
    )
    
    # dataset
    parser.add_argument(
        '--dataset',
        default='busi',
        help='dataset name'
    )
    parser.add_argument(
        '--data_dir',
        default='../inputs',
        help='dataset dir'
    )
    parser.add_argument(
        '--output_dir',
        default='../outputs',
        help='output dir'
    )
    
    # optimizer
    parser.add_argument(
        '--optimizer',
        default='AdamW',
        choices=['AdamW', 'Adam', 'SGD'],
        help='optim: ' + ' | '.join(['AdamW', 'Adam', 'SGD']) + '(default:AdamW)'
    )
    parser.add_argument(
        '--lr',
        '--learning_rate',
        default=0.0005,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='momentum'
    )
    parser.add_argument(
        '--weight_decay',
        default=1e-4,
        type=float,
        help='weight decay'
    )
    parser.add_argument(
        '--nesterov',
        default=False,
        type=str2bool,
        help='nesterov'
    )
    parser.add_argument(
        '--kan_lr',
        default=5e-4,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )
    parser.add_argument(
        '--kan_weight_decay',
        default=1e-5,
        type=float,
        help='weight decay'
    )

    # scheduler
    parser.add_argument(
        '--scheduler',
        default='CosineAnnealingLR',
        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR']
    )
    parser.add_argument(
        '--min_lr',
        default=1e-5,
        type=float,
        help='minimum learning rate'
    )
    parser.add_argument(
        '--factor',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--patience',
        default=4,
        type=int
    )
    parser.add_argument(
        '--milestones',
        default='50,75,100,125,200',
        type=str
    )
    parser.add_argument(
        '--gamma',
        default=2/3,
        type=float
    )
    parser.add_argument(
        '--early_stopping',
        default=-1,
        type=int,
        metavar='N',
        help='early stopping (default:-1)'
    )
    parser.add_argument(
        '--cfg',
        type=str,
        metavar="FILE",
        help='path to config file'
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int
    )
    parser.add_argument(
        '--no_kan',
        action='store_true'
    )

    config = parser.parse_args()
    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {
        'loss' : AverageMeter(),
        'iou' : AverageMeter(),
        'dice_coef_val' : AverageMeter(),
    }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            dice_coef_val = dice_coef(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            dice_coef_val = dice_coef(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice_coef_val'].update(dice_coef_val, input.size(0))

        postfix = OrderedDict(
            [
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice_coef_val', avg_meters['dice_coef_val'].avg),
            ]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict(
        [
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice_coef_val', avg_meters['dice_coef_val'].avg),
        ]
    )


def validate(config, val_loader, model, criterion):
    avg_meters = {
        'loss':AverageMeter(),
        'iou':AverageMeter(),
        'dice':AverageMeter(),
        'dice_coef_val':AverageMeter(),
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
                dice_coef_val = dice_coef(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)
                dice_coef_val = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['dice_coef_val'].update(dice_coef_val, input.size(0))

            postfix = OrderedDict(
            [
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('dice_coef_val', avg_meters['dice_coef_val'].avg),
            ]
        )
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict(
            [
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('dice_coef_val', avg_meters['dice_coef_val'].avg),
            ]
        )

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False # select a best algorithm
    torch.backends.cudnn.deterministic = True # use deterministic algorithms

def main():
    
    config = vars(parse_args())
    seed_torch(seed=config['dataseed'])
    #seed_torch(seed=2981)
    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_wnoDS' % (config['dataset'], config['arch'])

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    # print('-' * 20)
    # for key in config:
    #     print('%s : %s' % (key, config[key]))
    # print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f, sort_keys=True)

    # define loss function (criterion)
    if config['loss'] == 'FocalDiceLoss':
        criterion = losses.FocalDiceLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            dice_weight=config['dice_weight'],
            smooth=config['loss_smooth']
        ).cuda()
    elif config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    # create model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['in_chans'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan']
    )
    model = model.cuda()

    param_groups = []
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            # kan_fc_params.append(name)
            param_groups.append(
                {
                    'params':param,
                    'lr':config['kan_lr'],
                    'weight_decay':config['kan_weight_decay'],
                }
            )
        else:
            # other_params.append(name)
            param_groups.append(
                {
                    'params':param,
                    'lr':config['lr'],
                    'weight_decay':config['weight_decay'],
                }
            )
    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(param_groups, eps=1e-8, amsgrad=True)
    
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)

    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError
    
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr'],
        )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config['factor'],
            patience=config['patience'],
            verbose=True,
            min_lr=config['min_lr']
        )
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(e) for e in config['milestones'].split(',')
            ],
            gamma=config['gamma']
        )
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    
    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')
    
    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    #TODU

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    # dataseed
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        albu.RandomRotate90(),
        albu.HorizontalFlip(),
        #albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        #albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.2),
        #albu.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-45, 45), p=0.3),
        #albu.ElasticTransform(p=0.2),
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform,
    )

    if len(train_dataset) == 0:
        raise ValueError("train_dataset is empty!")


    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform,
    )

    if len(val_dataset) == 0:
        raise ValueError("val_dataset is empty!")


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
    )

    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise ValueError("train_loader is empty: no data yielded.") from None
    except Exception as e:
        raise RuntimeError(f"Failed to load a batch from train_loader: {str(e)}") from e

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
    )

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('train_loss', []),
        ('train_iou', []),
        ('train_dice_coef_val', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_dice_coef_val', []),
    ])
    
    best_iou = 0
    best_dice = 0
    #trigger = 0

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)

        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print(
            'train_loss %.4f - train_iou %.4f - val_loss %.4f - val_iou %.4f '
            % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'])
        )
        current_lr = optimizer.param_groups[0]['lr']
        log['epoch'].append(epoch)
        log['lr'].append(current_lr)
        log['train_loss'].append(train_log['loss'])
        log['train_iou'].append(train_log['iou'])
        log['train_dice_coef_val'].append(train_log['dice_coef_val'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_dice_coef_val'].append(val_log['dice_coef_val'])
        

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        # use tensorBoard for review
        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/dice_coef', val_log['dice_coef_val'], global_step=epoch)
        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        #trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('Best_IoU: %.4f' % best_iou)
            print('Best_Dice: %.4f' % best_dice)
            

            # early stopping
            # if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            #     print("=> early stopping")
            #     break

            torch.cuda.empty_cache()
    else:
        # close computer when for loop done
        # os.system("shutdown /s /t 3")
        pass

if __name__ == '__main__':
    main()
    


