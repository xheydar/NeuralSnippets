import os
from easydict import EasyDict as edict
import yaml
import numpy as np
from PIL import Image
import math
import cv2
import random
import platform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp

from mine.datasets import create_dataloader
from utils.general import labels_to_class_weights, labels_to_image_weights, init_seeds
from utils.autoanchor import check_anchors
from utils.torch_utils import ModelEMA
from utils.loss import ComputeLoss, ComputeLossOTA

from models.yolo import Model

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

class Train :
    def __init__( self, cfg ):
        self.cfg = cfg

        init_seeds(2 + cfg.rank)

        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name)

        with open(self.cfg.hyp, 'r') as ff :
            self.hyp = yaml.load( ff, Loader=yaml.SafeLoader)

        with open(self.cfg.data, 'r') as ff :
            self.data_dict = yaml.load( ff, Loader=yaml.SafeLoader)


    def load_datasets( self ):
        nc = int(self.data_dict['nc'])  # number of classes
        names = self.data_dict['names']  # class names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc)  # check

        train_dataloader, train_dataset = create_dataloader( self.cfg.datasets_path, 'train', self.cfg, self.hyp, self.data_dict )
        val_dataloader, val_dataset = create_dataloader( self.cfg.datasets_path, 'val', self.cfg, self.hyp, 
                                                         self.data_dict, rect=True, augment=False, pad=0.5 )


        self.datasets = {'train':train_dataset, 'val':val_dataset}
        self.dataloaders = {'train':train_dataloader, 'val':val_dataloader} 

    def load_model( self ):
        nc = int(self.data_dict['nc'])  # number of classes
        
        model = Model(self.cfg.model_cfg, ch=3, nc=nc, anchors=self.hyp.get('anchors')).to(self.device)

        freeze = [0]
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        total_batch_size = self.cfg.batch_size

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):           
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):           
                    pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):           
                    pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):           
                    pg0.append(v.imo.implicit)
                else:
                    for iv in v.imo:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):           
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)
            if hasattr(v, 'attn'):
                if hasattr(v.attn, 'logit_scale'):   
                    pg0.append(v.attn.logit_scale)
                if hasattr(v.attn, 'q_bias'):   
                    pg0.append(v.attn.q_bias)
                if hasattr(v.attn, 'v_bias'):  
                    pg0.append(v.attn.v_bias)
                if hasattr(v.attn, 'relative_position_bias_table'):  
                    pg0.append(v.attn.relative_position_bias_table)
            if hasattr(v, 'rbr_dense'):
                if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                    pg0.append(v.rbr_dense.weight_rbr_origin)
                if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                    pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                    pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                    pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                    pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(v.rbr_dense, 'vector'):   
                    pg0.append(v.rbr_dense.vector)

        if self.cfg.optimizer.adam:
            optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        del pg0, pg1, pg2

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR

        epochs = self.cfg.epochs

        if self.cfg.optimizer.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        else:
            lf = one_cycle(1, self.hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if self.cfg.rank in [-1, 0] else None


        # Image sizes
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])

        check_anchors(self.datasets['train'], model=model, thr=self.hyp['anchor_t'], imgsz=self.cfg.img_size['train'])
        model.half().float()

        # Model parameters
        self.hyp['box'] *= 3. / nl  # scale to layers
        self.hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        self.hyp['obj'] *= (self.cfg.img_size['train'] / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = 0.0 #opt.label_smoothing

        model.nc = nc  # attach number of classes to model
        model.hyp = self.hyp  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = labels_to_class_weights(self.datasets['train'].labels, nc).to(self.device) * nc  # attach class weights
        model.names = self.data_dict['names']

        self.model = {}
        self.model['model'] = model
        self.model['optimizer'] = optimizer
        self.model['scheduler'] = scheduler
        self.model['ema'] = ema
        self.model['lf'] = lf

        self.model['params'] = {}
        self.model['params']= gs
        self.model['nl'] = nl
        self.model['accumulate'] = accumulate

    def train( self ):

        cuda = self.device.type != 'cpu'
        dataloader = self.dataloaders['train']
        dataset = self.datasets['train']

        model = self.model['model']
        scheduler = self.model['scheduler']
        optimizer = self.model['optimizer']
        ema = self.model['ema']

        lf = self.model['lf']
        accumulate = self.model['accumulate']

        nbs = 64  # nominal batch size
        nc = int(self.data_dict['nc'])
        nb = len(dataloader)

        total_batch_size = self.cfg.batch_size

        epochs = self.cfg.epochs
        start_epoch, best_fitness = 0, 0.0
        nw = max(round(self.hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)

        print( nw )
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move

        scaler = amp.GradScaler(enabled=cuda)
        compute_loss_ota = ComputeLossOTA(model)  # init loss class
        compute_loss = ComputeLoss(model)  # init loss class

        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            model.train()

            if self.cfg.image_weights :
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

            mloss = torch.zeros(4, device=self.device) 
            pbar = enumerate(dataloader)
            pbar = tqdm(pbar, total=nb)

            for i, (imgs, targets, paths, _) in pbar:
                ni = i + nb * epoch
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
                targets = targets.to(self.device)

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])
                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    if 'loss_ota' not in self.hyp or self.hyp['loss_ota'] == 1:
                        loss, loss_items = compute_loss_ota(pred, targets, imgs)  # loss scaled by batch_size
                    else:
                        loss, loss_items = compute_loss(pred, targets)  # loss scaled by batch_size
                    if self.cfg.rank != -1:
                        loss *= opt.world_size  # gradient averaged between devices in DDP mode
                    if self.cfg.quad:
                        loss *= 4.
                
                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)


 
    def do_stuff( self ):
        img, label, file, size, label_raw = self.datasets['val'][150] 

        label = label.numpy()
        img = img.numpy().transpose([1,2,0]).astype( np.uint8 )
        img_h, img_w = img.shape[:2]

        img = np.ascontiguousarray(img, dtype=np.uint8)

        for l in label :
            x,y,w,h = l[2:]
            
            w = int(w * img_w)
            h = int(h * img_h)

            x = int(x * img_w - w/2)
            y = int(y * img_h - h/2)

        #    x0,y0,x1,y1 = l[1:].astype(np.int32)

            img = cv2.rectangle( img, (x,y,w,h), (255,0,0), 1 )

        img = Image.fromarray(img)
        img.show()

def load():
    cfg = edict()

    if platform.system() == "Darwin" :
        cfg.datasets_path = '/Users/heydar/Work/void/data/datasets'
    else :
        cfg.datasets_path = "/ssd/data/datasets"

    cfg.data = 'params/coco.yml'
    cfg.hyp = 'params/hyp.yml'
    cfg.rank = -1
    cfg.img_size = {'train':640, 'val':640}
    cfg.batch_size = 8
    cfg.grid_stride = 32
    cfg.world_size = 1
    cfg.image_weights = True
    cfg.epochs = 300
    cfg.dataloader = edict()
    cfg.dataloader.augment = True
    cfg.dataloader.cache_images = False
    cfg.dataloader.rect = False # Sorting the images by their aspect ratio
    cfg.dataloader.workers = 8
    cfg.dataloader.quad = False
    cfg.dataloader.pad = 0.0
    cfg.model_cfg = './cfg/training/yolov7.yaml'

    cfg.optimizer = edict()
    cfg.optimizer.adam = False
    cfg.optimizer.linear_lr = False

    cfg.model = edict()
    cfg.model.sync_bn = False

    #cfg.img_size.extend([cfg.img_size[-1]] * (2 - len(cfg.img_size)))

    train = Train( cfg )

    return train

if __name__=="__main__" :
    t = load()
    t.load_datasets()
    t.load_model()

    t.train()
