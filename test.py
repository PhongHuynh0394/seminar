import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import timm
import timm.optim.optim_factory as optim_factory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import random
from scipy.ndimage import gaussian_filter
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

def load_model_checkpoint(countmodel, checkpoint_path, do_resume=True):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle positional embedding mismatch
        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != \
                countmodel.model.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        # Load model weights
        countmodel.model.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % checkpoint_path)

        # Optionally load optimizer state and scaler for resuming training
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and do_resume:
            optimizer_state = checkpoint['optimizer']
            scaler_state = checkpoint.get('scaler', None)

            # Update the optimizer state in Lightning (manually inject state)
            optimizer = countmodel.configure_optimizers()
            if isinstance(optimizer, dict):
                optimizer = optimizer['optimizer']
            optimizer.load_state_dict(optimizer_state)

            if scaler_state and countmodel.loss_scaler is not None:
                countmodel.loss_scaler.load_state_dict(scaler_state)

            # Update starting epoch for training
            start_epoch = checkpoint['epoch'] + 1
            print("With optim & scheduler!")

from CounTR.util.misc import NativeScalerWithGradNormCount as NativeScaler
import CounTR.models_mae_cross as models_mae_cross

class CountingModel(LightningModule):
    def __init__(self,
                 model="mae_vit_base_patch16",
                 pretain_weight=None,
                 device=None,
                 shot_num=3):
        super().__init__()
        self.model = models_mae_cross.__dict__[model](norm_pix_loss=False)

        if pretain_weight:
          load_model_checkpoint(self, pretain_weight)

        self.device = device
        self.loss_scaler = NativeScaler()
        self.save_hyperparameters()

    def forward(self, samples, boxes, shot_num):
        return self.model(samples, boxes, shot_num)

    def training_step(self, batch, batch_idx):
        samples, boxes, gt_density, m_flag = batch['image'], batch['boxes'], batch['density_gt'], batch['m_flag']
        device = self.device

        # If there is at least one image in the batch using Type 2 Mosaic, 0-shot is banned.
        flag = sum(m_flag)
        shot_num = random.randint(1, 3) if flag else random.randint(0, 3)

        # Forward pass
        output = self(samples, boxes, shot_num)

        # Compute loss
        mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
        masks = torch.from_numpy(np.tile(mask, (output.shape[0], 1)).reshape(output.shape[0], 384, 384)).to(device)
        loss = ((output - gt_density) ** 2 * masks / (384 * 384)).sum() / output.shape[0]

        # MAE and RMSE
        pred_cnt = output.view(len(samples), -1).sum(1) / 60
        gt_cnt = gt_density.view(len(samples), -1).sum(1) / 60
        mae = torch.abs(pred_cnt - gt_cnt).mean()
        rmse = (torch.abs(pred_cnt - gt_cnt) ** 2).mean() ** 0.5

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        self.log("train_rmse", rmse, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, boxes, gt_density, m_flag = batch['image'], batch['boxes'], batch['density_gt'], batch['m_flag']
        device = self.device

        # Forward pass
        shot_num = random.randint(0, 3)
        output = self(samples, boxes, shot_num)

        # MAE, RMSE, and NAE
        pred_cnt = output.view(len(samples), -1).sum(1) / 60
        gt_cnt = gt_density.view(len(samples), -1).sum(1) / 60
        cnt_err = torch.abs(pred_cnt - gt_cnt).float()
        mae = cnt_err.mean()
        rmse = (cnt_err ** 2).mean() ** 0.5
        nae = (cnt_err / gt_cnt).mean()

        self.log("val_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_nae", nae, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        samples, boxes, gt_density, m_flag = batch['image'], batch['boxes'], batch['density_gt'], batch['m_flag']
        device = self.device

        # Forward pass
        shot_num = random.randint(0, 3)
        output = self(samples, boxes, shot_num)

        # MAE, RMSE, and NAE
        pred_cnt = output.view(len(samples), -1).sum(1) / 60
        gt_cnt = gt_density.view(len(samples), -1).sum(1) / 60
        cnt_err = torch.abs(pred_cnt - gt_cnt).float()
        mae = cnt_err.mean()
        rmse = (cnt_err ** 2).mean() ** 0.5
        nae = (cnt_err / gt_cnt).mean()

        self.log("test_mae", mae, prog_bar=True)
        self.log("test_rmse", rmse, prog_bar=True)
        self.log("test_nae", nae, prog_bar=True)

    def configure_optimizers(self):
        eff_batch_size = self.args.batch_size * self.args.accum_iter
        lr = self.args.blr * eff_batch_size / 256 if self.args.lr is None else self.args.lr
        param_groups = optim_factory.add_weight_decay(self.model, self.args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer
