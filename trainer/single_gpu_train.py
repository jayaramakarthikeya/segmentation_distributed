
import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from base_trainer import BaseTrainer

class SingleGPUTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.device = self.config['device']
        self.model = model.to(self.device)

        self.num_classes = self.train_loader.dataset.num_classes

        super(SingleGPUTrainer,self).__init__(config, model, train_loader, val_loader)

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        self.log_step = self.config['trainer']['tensorboard_log_step']
        

    def _train_epoch(self,epoch):

        self.writer_mode = 'train'
        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)

        for batch_idx , (images,labels) in enumerate(self.train_loader):
            self.data_time.update(time.time() - tic)
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
                assert output.size()[2:] == labels.size()[1:]
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