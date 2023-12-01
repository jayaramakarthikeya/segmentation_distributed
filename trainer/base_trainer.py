import sys
sys.path.append('../')
import torch
import numpy as np
import datetime
import os
import utils.helpers as helpers
from torch.utils import tensorboard
import json
import math
from trainer.early_stopping import EarlyStopping

from utils import losses
import time
from tqdm import tqdm
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from torchvision.utils import make_grid
from torchvision import transforms

class BaseTrainer:
    def __init__(self,config,model,train_loader,val_loader,logger):
        self.logger = logger
        self.train_loader = train_loader
        self.config = config
        self.model = model
        trainable_params = filter(lambda p:p.requires_grad, self.model.parameters())
        self.optimizer = getattr(torch.optim, config['optimizer']['type'])(trainable_params, **config['optimizer']['args'])
        self.val_loader = val_loader
        self.model_type = self.model.model_type
        lr_sheduler_config = None
        if lr_sheduler_config is not None:
            self.lr_sheduler = getattr(torch.optim.lr_scheduler, lr_sheduler_config['type'])(self.optimizer,**lr_sheduler_config['args'])
        else:
            self.lr_sheduler = None
        
        self.loss = getattr(losses, config['loss'])()
        

        self.writer_mode = 'train'

        #CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        torch.backends.cudnn.benchmark = True

        #MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

        #CHECKPOINTS & TENSOBOARD
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['name'], start_time)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(cfg_trainer['log_dir'], self.config['name'], start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

        #Early Stopping
        self.early_stoping = EarlyStopping(self.model,self.model_type,self.optimizer,self.config,self.checkpoint_dir,self.mnt_mode,trace_func=self.logger)

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        self.log_step = self.config['trainer']['tensorboard_log_step']

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus
    

    def train(self):

        self.model.summary()
        for epoch in range(self.epochs):
            results = self._train_epoch(epoch)
            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)

            self.logger.info(f"Results for {epoch} epoch: ")
            for k , v in results.items():
                self.logger.info(f" {str(k)}: {v}")
            

            self.early_stoping(results[self.mnt_metric],epoch,self.model)

            if self.early_stoping.early_stop:
                self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping.counter} epochs')
                self.logger.warning('Training Stopped')
                break


        

    def _train_epoch(self, epoch):
        self.writer_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)

        for batch_idx , (images,labels) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            if self.lr_sheduler is not None:
                self.lr_sheduler.step(epoch=epoch-1)

            #FORWARD PASS
            self.optimizer.zero_grad()
            output = self.model(images)

            #BACKWARD PASS AND OPTIMIZE
            if self.model.model_type[:3] == "PSP":
                assert output[0].size()[2:] == labels.size()[1:]
                assert output[0].size()[1] == self.num_classes 
                loss = self.loss(output[0], labels)
                loss += self.loss(output[1], labels) * 0.4
                output = output[0]
            else:
                assert output.size()[2:] == labels.size()[2:]
                assert output.size()[1] == self.num_classes 
                loss = self.loss(output, labels)

            
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.writer_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, labels, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()


            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average, 
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        seg_metrics = self._get_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]: 
            self.writer.add_scalar(f'{self.writer_mode}/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.writer_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.writer_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (images,labels) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                
                output = self.model(images)
                loss = self.loss(output, labels)
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, labels, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = labels.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([images[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        #print(inter,union)
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }