import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from utilities.save_helper import get_checkpoint_state
from utilities.save_helper import load_checkpoint
from utilities.save_helper import save_checkpoint
from utilities.loss.centernet_loss import compute_centernet3d_loss


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.max_objs = 50    # max objects per images, defined in dataset
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # loading pretrain/resume model
        if cfg['pretrained']:
            load_checkpoint(model=self.model,
                            optimizer=None,
                            map_location=self.device,
                            logger=self.logger)
            print('load pretrain model ')

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids).to(self.device)



    def train(self):
        start_epoch = self.epoch
        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(self.epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()


            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_name = os.path.join('checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

            progress_bar.update()

        return None


    def train_one_epoch(self, epoch):
        self.model.train()
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        iou_loss =[]
        other_loss = []
        for batch_idx, (inputs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)


            calibs = [self.train_loader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}


            #print('img id: ', info['img_id'])
            #plt.imshow(inputs[0,:,:,:].cpu().numpy().transpose(1, 2, 0))
            #plt.show()

            cls_mean_size = self.train_loader.dataset.cls_mean_size

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            total_loss, stats_batch = compute_centernet3d_loss(outputs, targets, self.max_objs, calibs, info, cls_mean_size, threshold=0.2)
            iou_loss.append(stats_batch['iou_loss'])
            other_loss.append(stats_batch['seg'] + stats_batch['offset2d'] + stats_batch['size2d'] + stats_batch['offset3d'] + stats_batch['depth'] + stats_batch['size3d'] + stats_batch['heading'])


            total_loss.backward()
            self.optimizer.step()

            progress_bar.update()

        my_array = np.array(iou_loss)
        my_array2 = np.array(other_loss)
        file_path_iou = "/Users/strom/Desktop/monodle/loss_info/"+str(epoch)+"iouloss.npy"
        file_path_other = "/Users/strom/Desktop/monodle/loss_info/"+str(epoch)+"otherloss.npy"

        np.save(file_path_iou, my_array)
        np.save(file_path_other, my_array2)



        progress_bar.close()




