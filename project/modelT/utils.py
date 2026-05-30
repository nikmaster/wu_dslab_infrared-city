#This notebook is based on the code of:
#V. S. F. Garnot and L. Landrieu, “Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4872–4881, doi: 10.1109/ICCV48922.2021.00483.

import collections.abc
import re
import os
import time
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch.nn.init as init

np_str_obj_array_pattern = re.compile(r"[SaUO]")

@dataclass
class Config:
    # Model parameters
    input_dim = 5
    encoder_widths: list = field(default_factory=lambda: [32, 32, 64, 64])
    decoder_widths: list = field(default_factory=lambda: [32, 32, 64, 64])

    # convolution
    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1

    # aggregation and normalization
    agg_mode: str = "att_group"
    encoder_norm: str = "batch"

    # attention
    n_head: int = 4
    d_model: int = 32
    d_k: int = 8

    # Padding
    pad_value: float = 0.0
    padding_mode: str = "reflect"

    # workers and device
    num_workers: int = 4
    rdm_seed: int = 1
    cache: bool = False
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    def torch_device(self):
        return torch.device(self.device)
    
    # loss
    pos_weight = 1
    presence_threshold = 0.5
    count_loss_weight = 2
    count_loss_negative_mask = 0.005

    # Training
    epochs: int = 1
    batch_size: int = 4
    lr: float = 3e-4

    # Validation and display
    val_every: int = 1
    val_after: int = 0
    display_step: int = 5

    # Folds
    fold: int = 1


def pad_tensor(x, l, pad_value=0):
    # This function pads a tensor
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def pad_collate(batch, pad_value=0):
    # This function helps pad batches on the time dimension for the data loader
    # Modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py

    # Get the first element of the batch to determine its type
    elem = batch[0]
    elem_type = type(elem)

    # Check if batch elements are PyTorch tensors
    if isinstance(elem, torch.Tensor):
        out = None

        # Check sequence lengths along first dimension
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)

            # Pad tensors if lengths differ
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]

        # Use shared memory when using DataLoader workers
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

         # Stack tensors into a batch
        return torch.stack(batch, 0, out=out)
    
    # Check if batch elements are NumPy arrays
    elif (elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_"):

        # Convert NumPy arrays to tensors
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":

            # Reject string/object arrays
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        
        # Handle NumPy scalars
        elif elem.shape == ():
            return torch.as_tensor(batch)
    
    # Check if batch elements are dictionaries
    elif isinstance(elem, collections.abc.Mapping):

        # Recursively collate dictionary values
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    
    # Check if batch elements are named tuples
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    
    # Check if batch elements are lists or sequences
    elif isinstance(elem, collections.abc.Sequence):

        # Ensure all sequences have same length
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        
        # Transpose and collate recursively
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    # Unsupported type
    raise TypeError(f"Format not managed : {elem_type}")


def get_ntrainparams(model):
    # This function gets the number of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TreeDataset(Dataset):
    # Generates the combined dataset with inputs, labels, dates and which pixels should be used for training

    def __init__(self, patches, trees, dates, usables = None, augment=False):
        # Initialize all values
        self.patches = patches
        self.trees = trees
        self.dates = dates
        self.usables = usables
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):

        # Load sample
        x = self.patches[idx]
        y = self.trees[idx]
        date = self.dates[idx]

        # Get the training mask
        train_mask = None
        if self.usables is not None:
            train_mask = self.usables[idx]

        # Random spatial transform on data
        if self.augment:

            # Random rotation
            k = np.random.choice([0, 1, 2, 3])
            x = np.rot90(x, k=k, axes=(-2, -1)).copy()
            y = np.rot90(y,k=k).copy()
            if train_mask is not None:
                train_mask = np.rot90(train_mask,k=k).copy()

            # Random horizontal flip
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=-1).copy()
                y = np.flip(y, axis=-1).copy()
                if train_mask is not None:
                    train_mask = np.flip(train_mask, axis=-1).copy()

            # Random vertical flip
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=-2).copy()
                y = np.flip(y, axis=-2).copy()
                if train_mask is not None:
                    train_mask = np.flip(train_mask, axis=-2).copy()

        # Convert everything to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        date = torch.tensor(date, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        # If mask exists, convert and return full tuple
        if train_mask is not None:
            train_mask = torch.tensor(train_mask, dtype=torch.float32).unsqueeze(0)

            return x, y, date, train_mask

        # Otherwise return without mask
        return x, y, date

def recursive_todevice(x, device):
    # This function puts different data types to correct device

    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

class AverageMeter:
    # This function gets the average of metrics

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def add(self, val):
        self.sum += val
        self.count += 1

    def value(self):
        return self.sum / max(self.count, 1)

def iterate(model, data_loader, config_train, config_test = None, optimizer=None, mode="train", img_output = False):
    # Iterates over each batch in the data_loader and trains or evaluates the model

    # Initialize metric trackers
    loss_meter = AverageMeter()
    loss_presence_meter = AverageMeter()
    loss_count_meter = AverageMeter()
    mae_meter = AverageMeter()
    rmse_meter = AverageMeter()
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    pred_pos_meter = AverageMeter()
    true_pos_meter = AverageMeter()

    # Storage for predictions
    all_imgs = []
    all_probs = []
    all_preds = []
    all_targets = []

    # Select device and logging frequency
    if config_test is None:
        device = config_train.device
        display_step = config_train.display_step
    else:
        device = config_test.device
        display_step = config_test.display_step

    t_start = time.time()

    # Iteration over batches
    for i, batch in enumerate(data_loader):

        # Move batch to device
        batch = recursive_todevice(batch, device)
        
        # Unpack batch (with or without mask)
        if len(batch) == 4:
            x, y, dates, valid_mask = batch
            valid_mask = valid_mask.float()
        else:
            x, y, dates = batch
            valid_mask = torch.ones_like(y)
        y = y.float()

        if img_output:
            all_imgs.append(x.detach().cpu().numpy())

        # Forward pass
        if mode != "train":
            with torch.no_grad():
                presence_logits, lambda_pred = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            presence_logits, lambda_pred = model(x, batch_positions=dates)
            
        # Calculate presence prediction target
        presence_target = (y > 0).float()
        
        # Presence loss (weighted BCE)
        presence_loss = F.binary_cross_entropy_with_logits(presence_logits,presence_target,weight=(presence_target * config_train.pos_weight + (1 - presence_target)) * valid_mask,reduction='sum') / (valid_mask.sum() + 1e-6)

        # Count loss (log-space smooth L1)
        positive_mask = (y > 0).float()
        negative_mask = (y == 0).float()
        weights = (positive_mask + negative_mask * config_train.count_loss_negative_mask) * valid_mask
        #pred = torch.log1p(torch.expm1(lambda_pred).clamp(min=0))
        count_pred = F.softplus(lambda_pred)
        #count_loss = (weights * F.smooth_l1_loss(pred, torch.log1p(y), reduction='none')).sum() / (weights.sum() + 1e-6)
        count_loss = (weights * F.smooth_l1_loss(torch.log1p(count_pred),torch.log1p(y),reduction='none')).sum() / (weights.sum() + 1e-6)


        # Total loss
        loss = presence_loss + config_train.count_loss_weight * count_loss

        # Final metrics
        with torch.no_grad():
            presence_prob = torch.sigmoid(presence_logits)

            # Final prediction
            pred_count = (presence_prob > config_train.presence_threshold).float() * count_pred
            #pred_count = (presence_prob > config_train.presence_threshold).float() * torch.expm1(lambda_pred)
            
            # Regression metrics
            mask = (y > 0) & (valid_mask > 0)
            mae_masked = (mask * torch.abs(pred_count - y)).sum() / (mask.sum() + 1e-6)
            rmse = torch.sqrt(torch.mean((pred_count - y) ** 2))

            # Classification metrics
            pred_presence = (presence_prob > config_train.presence_threshold).float()
            acc = (((pred_presence == presence_target).float() * valid_mask).sum() / (valid_mask.sum() + 1e-6))
            valid_bool = valid_mask > 0
            tp = ((pred_presence == 1)& (presence_target == 1)& valid_bool).sum()
            fp = ((pred_presence == 1)& (presence_target == 0)& valid_bool).sum()
            fn = ((pred_presence == 0)& (presence_target == 1)& valid_bool).sum()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            pred_pos = pred_presence.mean().item()
            true_pos = presence_target.mean().item()

            # Save for later analysis
            pred = pred_count.detach().cpu().numpy()
            true = y.detach().cpu().numpy()
            all_preds.append(pred)
            all_targets.append(true)
            all_probs.append(presence_prob)

        # Backpropagation
        if mode == "train":
            loss.backward()
            optimizer.step()

        # Update running metrics
        mae_meter.add(mae_masked.item())
        rmse_meter.add(rmse.item())
        loss_meter.add(loss.item())
        loss_presence_meter.add(presence_loss.item())
        loss_count_meter.add(count_loss.item())
        acc_meter.add(acc.item())
        precision_meter.add(precision.item())
        recall_meter.add(recall.item())
        pred_pos_meter.add(pred_pos)
        true_pos_meter.add(true_pos)

        # Displaying metrics
        if (i + 1) % display_step == 0:
            print(f"Step [{i + 1}/{len(data_loader)}] epoch values, Loss: {loss_meter.value():.4f}, Presence loss: {loss_presence_meter.value():.4f}, "
                  f"Count loss: {loss_count_meter.value():.4f}, RMSE: {rmse_meter.value():.4f}, MAE masked: {mae_meter.value():.4f}, "
                  f"Accuracy: {acc_meter.value():.4f}, Precision: {precision_meter.value():.4f}, Recall: {recall_meter.value():.4f}, "
                  f"True positives: {true_pos_meter.value():.4f}, Predicted positives: {pred_pos_meter.value():.4f}")
            
            print(f"Step [{i + 1}/{len(data_loader)}] batch values, Loss: {loss:.4f}, Presence loss: {presence_loss:.4f}, "
                  f"Count loss: {count_loss:.4f}, RMSE: {rmse:.4f}, MAE masked: {mae_masked:.4f}, "
                  f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"True positives: {true_pos:.4f}, Predicted positives: {pred_pos:.4f}")
            
    # Epoch summary
    t_end = time.time()
    total_time = t_end - t_start
    print(f"Epoch time: {total_time:.1f}")
    metrics = {
        f"{mode}_loss": loss_meter.value(),
        f"{mode}_presence_loss": loss_presence_meter.value(),
        f"{mode}_count_loss": loss_count_meter.value(),
        f"{mode}_rmse": rmse_meter.value(),
        f"{mode}_mae_masked": mae_meter.value(),
        }

    if img_output:
        return metrics, all_preds, all_targets, all_probs, all_imgs
    else: 
        return metrics, all_preds, all_targets

def prepare_output(config_paths, config):
    # This function prepares output directories to store the results

    # Ensure main results directory exists
    os.makedirs(config_paths.res_dir, exist_ok=True)

    # Create all needed directories for folds
    if config.fold is None:
        for fold in range(1, 6):
            os.makedirs(os.path.join(config_paths.res_dir, f"Fold_{fold}"), exist_ok=True)
    else:
        os.makedirs(os.path.join(config_paths.res_dir, f"Fold_{config.fold}"), exist_ok=True)

def config_to_json(config):
    # This function recursively converts a configuration object into JSON-serializable format

    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    return convert(asdict(config))

def save_results(fold, metrics=None, config=None, config_paths=None, preds=None, targets=None, probs = None, imgs = None):
    # This function saves the final results

    out_dir = os.path.join(config_paths.res_dir, f"Fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics (MAE, RMSE, R2)
    if metrics is not None:
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    # Save predictions and ground truth 
    if preds is not None:
        np.save(os.path.join(out_dir, "predictions.npy"), np.concatenate(preds, axis=0).astype(np.float32))
    if targets is not None:
        np.save(os.path.join(out_dir, "targets.npy"), np.concatenate(targets, axis=0).astype(np.float32))

    # Save input images
    if imgs is not None:
        np.save(os.path.join(out_dir, "images.npy"), np.concatenate(imgs, axis=0).astype(np.float32))
    
    # Save the predicted probabilities
    if probs is not None:
        np.save(os.path.join(out_dir, "probabilities.npy"), np.concatenate(probs, axis=0).astype(np.float32))

    # Save configs for reproducibility
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_to_json(config), f, indent=4)

    with open(os.path.join(out_dir, "config_paths.json"), "w") as f:
        json.dump(config_to_json(config_paths), f, indent=4)

def weight_init(m):
    # This function initializes neural network weights depending on layer type

    # Convolutional layers
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.ConvTranspose1d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    # Linear layers
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    # Normalization layers
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)#

def load_config(config_test):
    # load the config used for training

    with open(config_test.config_path, "r") as f:
        cfg_dict = json.load(f)

    config_train = Config()

    for k, v in cfg_dict.items():
        if hasattr(config_train, k):
            try:
                if isinstance(getattr(config_train, k), Path):
                    v = Path(v)
            except:
                pass
            setattr(config_train, k, v)

    return config_train