# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# IMPORTS
import pprint
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim.lr_scheduler as scheduler
import yacs.config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from FastSurferCNN.config.global_var import get_class_names
from FastSurferCNN.data_loader import loader
from FastSurferCNN.models.losses import get_loss_func
from FastSurferCNN.models.network_mae import build_model
from FastSurferCNN.models.optimizer import get_optimizer
from FastSurferCNN.utils import checkpoint as cp
from FastSurferCNN.utils import logging
from FastSurferCNN.utils.lr_scheduler import get_lr_scheduler
from FastSurferCNN.utils.meters import Meter
from FastSurferCNN.utils.metrics import iou_score, precision_recall
from FastSurferCNN.utils.misc import plot_predictions, update_num_steps

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for the networks.

    Methods
    -------
    __init__
        Construct object.
    train
        Trains the network.
    eval
        Validates calculations.
    run
        Performs training loop.
    """

    def __init__(self, cfg: yacs.config.CfgNode):
        """
        Construct Trainer object.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            Node of configs to be used.
        """
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Create the checkpoint dir.
        self.checkpoint_dir = cp.create_checkpoint_dir(cfg.LOG_DIR, cfg.EXPR_NUM)
        
        logging.setup_logging(os.path.join(cfg.LOG_DIR, "logs", cfg.EXPR_NUM + ".log"))
        logger.info("Training with config:")
        logger.info(pprint.pformat(cfg))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg)
        self.model.initialize_encoder_from_pretrained(cfg.MAE_MODEL)
        self.loss_func = get_loss_func(cfg)

        # set up class names
        self.class_names = get_class_names(cfg.DATA.PLANE, cfg.DATA.CLASS_OPTIONS)

        # Set up logger format
        self.a = "{}\t" * (cfg.MODEL.NUM_CLASSES - 2) + "{}"
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.plot_dir = os.path.join(cfg.LOG_DIR, "pred", str(cfg.EXPR_NUM))
        os.makedirs(self.plot_dir, exist_ok=True)

        self.subepoch = False if self.cfg.TRAIN.BATCH_SIZE == 16 else True

    def train(
        self,
        train_loader: loader.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: None | scheduler.StepLR | scheduler.CosineAnnealingWarmRestarts,
        train_meter: Meter,
        epoch,
    ) -> None:
        """
        Train the network to the given training data.

        Parameters
        ----------
        train_loader : loader.DataLoader
            Data loader for the training.
        optimizer : torch.optim.Optimizer
            Optimizer for the training.
        scheduler : None, scheduler.StepLR, scheduler.CosineAnnealingWarmRestarts
            LR scheduler for the training.
        train_meter : Meter
            Meter to keep track of the training stats.
        epoch : int
            Current epoch.

        """
        self.model.train()
        logger.info("Training started ")
        epoch_start = time.time()
        loss_batch = np.zeros(1)

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels, weights, scale_factors = (
                batch["image"].to(self.device),
                batch["label"].to(self.device),
                batch["weight"].float().to(self.device),
                batch["scale_factor"],
            )

            if not self.subepoch or (curr_iter) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.zero_grad()  # every second epoch to get batchsize of 16 if using 8

            pred = self.model(images, scale_factors)
            weights = None
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            train_meter.update_stats(pred, labels, loss_total)
            train_meter.log_iter(curr_iter, epoch)
            if scheduler is not None:
                train_meter.write_summary(
                    loss_total, scheduler.get_lr(), loss_ce, loss_dice
                )
            else:
                train_meter.write_summary(
                    loss_total, [self.cfg.OPTIMIZER.BASE_LR], loss_ce, loss_dice
                )

            loss_total.backward()
            if (
                not self.subepoch
                or (curr_iter + 1) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0
            ):
                optimizer.step()  # every second epoch to get batchsize of 16 if using 8
                if scheduler is not None:
                    scheduler.step(epoch + curr_iter / len(train_loader))

            loss_batch += loss_total.item()

            # Plot sample predictions
            if curr_iter == len(train_loader) - 2:
                plt_title = "Training Results Epoch " + str(epoch)

                file_save_name = os.path.join(
                    self.plot_dir, "Epoch_" + str(epoch) + "_Training_Predictions.pdf"
                )

                _, batch_output = torch.max(pred, dim=1)
                plot_predictions(
                    images, labels, batch_output, plt_title, file_save_name
                )

        train_meter.log_epoch(epoch)
        logger.info(
            f"Training epoch {epoch} finished in {time.time() - epoch_start:.04f} seconds"
        )

    @torch.no_grad()
    def eval(
        self, val_loader: loader.DataLoader, val_meter: Meter, epoch: int
    ) -> np.ndarray:
        """
        Evaluate model and calculates stats.

        Parameters
        ----------
        val_loader : loader.DataLoader
            Value loader.
        val_meter : Meter
            Meter for the values.
        epoch : int
            Epoch to evaluate.

        Returns
        -------
        int, float, ndarray
            median miou [value].
        """
        logger.info(f"Evaluating model at epoch {epoch}")
        self.model.eval()

        val_loss_total = defaultdict(float)
        val_loss_dice = defaultdict(float)
        val_loss_ce = defaultdict(float)

        ints_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        unis_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        miou = np.zeros(self.num_classes - 1)
        per_cls_counts_gt = defaultdict(lambda: np.zeros(self.num_classes - 1))
        per_cls_counts_pred = defaultdict(lambda: np.zeros(self.num_classes - 1))
        accs = defaultdict(
            lambda: np.zeros(self.num_classes - 1)
        )  # -1 to exclude background (still included in val loss)

        val_start = time.time()
        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, labels, weights, scale_factors = (
                batch["image"].to(self.device),
                batch["label"].to(self.device),
                batch["weight"].float().to(self.device),
                batch["scale_factor"],
            )

            pred = self.model(images, scale_factors)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)
            _, batch_output = torch.max(pred, dim=1)

            sf = torch.unique(scale_factors)
            if len(sf) != 1:
                continue  # skip batches with mixed scale_factors
            if len(sf) == 1:
                sf = sf.item()
                val_loss_total[sf] += loss_total.item()
                val_loss_dice[sf] += loss_dice.item()
                val_loss_ce[sf] += loss_ce.item()

                _, batch_output = torch.max(pred, dim=1)

                # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
                int_, uni_ = iou_score(batch_output, labels, self.num_classes)
                ints_[sf] += int_
                unis_[sf] += uni_

                tpos, pcc_gt, pcc_pred = precision_recall(
                    batch_output, labels, self.num_classes
                )
                accs[sf] += tpos
                per_cls_counts_gt[sf] += pcc_gt
                per_cls_counts_pred[sf] += pcc_pred

            # Plot sample predictions
            if curr_iter == (len(val_loader) // 2):
                plt_title = "Validation Results Epoch " + str(epoch)

                file_save_name = os.path.join(
                    self.plot_dir,
                    "Epoch_" + str(epoch) + "_Validations_Predictions.pdf",
                )

                plot_predictions(
                    images, labels, batch_output, plt_title, file_save_name
                )

            val_meter.update_stats(pred, labels, loss_total)
            val_meter.write_summary(loss_total)
            val_meter.log_iter(curr_iter, epoch)

        val_meter.log_epoch(epoch)
        logger.info(
            f"Validation epoch {epoch} finished in {time.time() - val_start:.04f} seconds"
        )

        # Get final measures and log them
        for key in accs.keys():
            ious = ints_[key] / unis_[key]
            miou += ious
            val_loss_total[key] /= curr_iter + 1
            val_loss_dice[key] /= curr_iter + 1
            val_loss_ce[key] /= curr_iter + 1

            # Log metrics
            logger.info(
                f"[Epoch {epoch} stats]: SF: {key}, MIoU: {np.mean(ious):.4f}; "
                f"Mean Recall: {np.mean(accs[key] / per_cls_counts_gt[key]):.4f}; "
                f"Mean Precision: {np.mean(accs[key] / per_cls_counts_pred[key]):.4f}; "
                f"Avg loss total: {val_loss_total[key]:.4f}; "
                f"Avg loss dice: {val_loss_dice[key]:.4f}; "
                f"Avg loss ce: {val_loss_ce[key]:.4f}"
            )

            logger.info(self.a.format(*self.class_names))
            logger.info(self.a.format(*ious))

        return np.mean(np.mean(miou))

    def run(self):
        """
        Transfer the model to devices, create a tensor board summary writer and then perform the training loop.
        """
        if self.cfg.NUM_GPUS > 1:
            assert (
                self.cfg.NUM_GPUS <= torch.cuda.device_count()
            ), "Cannot use more GPU devices than available"
            print("Using ", self.cfg.NUM_GPUS, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        val_loader = loader.get_dataloader(self.cfg, "val")
        train_loader = loader.get_dataloader(self.cfg, "train")

        update_num_steps(train_loader, self.cfg)

        # Transfer the model to device(s)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_lr_scheduler(optimizer, self.cfg)

        checkpoint_paths = cp.get_checkpoint_path(
            self.cfg.LOG_DIR, self.cfg.TRAIN.RESUME_EXPR_NUM
        )
        if self.cfg.TRAIN.RESUME and checkpoint_paths:
            try:
                checkpoint_path = checkpoint_paths.pop()
                checkpoint_epoch, best_metric = cp.load_from_checkpoint(
                    checkpoint_path,
                    self.model,
                    optimizer,
                    scheduler,
                    self.cfg.TRAIN.FINE_TUNE,
                )
                start_epoch = checkpoint_epoch
                best_miou = best_metric
                logger.info(f"Resume training from epoch {start_epoch}")
            except Exception as e:
                print(
                    f"No model to restore. Resuming training from Epoch 0. {e}"
                )
        else:
            logger.info("Training from scratch")
            start_epoch = 0
            best_miou = 0

        logger.info(
            f"{sum(x.numel() for x in self.model.parameters())} parameters in total"
        )

        # Create tensorboard summary writer

        writer = SummaryWriter(self.cfg.SUMMARY_PATH, flush_secs=15)

        train_meter = Meter(
            self.cfg,
            mode="train",
            global_step=start_epoch * len(train_loader),
            total_iter=len(train_loader),
            total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
            device=self.device,
            writer=writer,
        )

        val_meter = Meter(
            self.cfg,
            mode="val",
            global_step=start_epoch,
            total_iter=len(val_loader),
            total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
            device=self.device,
            writer=writer,
        )

        logger.info(f"Summary path {self.cfg.SUMMARY_PATH}")
        # Perform the training loop.
        logger.info(f"Start epoch: {start_epoch + 1}")

        for epoch in range(start_epoch, self.cfg.TRAIN.NUM_EPOCHS):
            self.train(train_loader, optimizer, scheduler, train_meter, epoch=epoch)

            if epoch % 10 == 0:
                val_meter.enable_confusion_mat()
                miou = self.eval(val_loader, val_meter, epoch=epoch)
                val_meter.disable_confusion_mat()

            else:
                miou = self.eval(val_loader, val_meter, epoch=epoch)

            if (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                logger.info(f"Saving checkpoint at epoch {epoch+1}")
                cp.save_checkpoint(
                    self.checkpoint_dir,
                    epoch + 1,
                    best_miou,
                    self.cfg.NUM_GPUS,
                    self.cfg,
                    self.model,
                    optimizer,
                    scheduler,
                )

            if miou > best_miou:
                best_miou = miou
                logger.info(
                    f"New best checkpoint reached at epoch {epoch+1} with miou of {best_miou}\nSaving new best model."
                )
                cp.save_checkpoint(
                    self.checkpoint_dir,
                    epoch + 1,
                    best_miou,
                    self.cfg.NUM_GPUS,
                    self.cfg,
                    self.model,
                    optimizer,
                    scheduler,
                    best=True,
                )
