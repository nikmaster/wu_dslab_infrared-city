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
from torch.nn import functional as F
from torch.utils.data import Dataset

np_str_obj_array_pattern = re.compile(r"[SaUO]")

class Config:
    # Model parameters
    input_dim = 5
    encoder_widths: list = field(default_factory=lambda: [16, 32, 32, 32])
    decoder_widths: list = field(default_factory=lambda: [16, 16, 32, 32])
    out_conv: int = 1

    str_conv_k: int = 4
    str_conv_s: int = 2
    str_conv_p: int = 1

    agg_mode: str = "att_group"
    encoder_norm: str = "group"

    n_head: int = 4
    d_model: int = 32
    d_k: int = 4

    num_workers: int = 4
    rdm_seed: int = 1
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    def torch_device(self):
        return torch.device(self.device)

    cache: bool = False

    # Training
    epochs: int = 5
    batch_size: int = 4
    lr: float = 0.001

    num_classes: int = 20
    ignore_index: int = -1
    pad_value: float = 0.0
    padding_mode: str = "reflect"

    val_every: int = 1
    val_after: int = 0
    display_step: int = 1

    fold: int = 1

# to pad a tensor
def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

# to call the data loader 
def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TreeDataset(Dataset):
    # generates the combined dataset with inputs, dates and labels

    def __init__(self, patches, dates, masks):
        """
        patches: (N, T, C, H, W)
        dates:   (N, T)
        masks:   (N, H, W)
        """
        self.patches = patches
        self.dates = dates
        self.masks = masks


    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]
        y = self.masks[idx]
        date = self.dates[idx]

        # convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        date = torch.tensor(date, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, date, y

def recursive_todevice(x, device):
    # put data to right device
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

class AverageMeter:
    # get the average of metrics

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

def iterate(model, data_loader, criterion, config, optimizer=None, mode="train", img_output = False):
    # iterates over each batch in the data_loader and trains/tests model

    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    rmse_meter = AverageMeter()
    all_imgs = []
    all_preds = []
    all_targets = []

    t_start = time.time()
    for i, batch in enumerate(data_loader):
        batch = recursive_todevice(batch, config.device)
        x, dates, y = batch
        y = y.float()
        if img_output:
            all_imgs.append(x.detach().cpu().numpy())

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates)
        else:
            optimizer.zero_grad()
            out = model(x, batch_positions=dates)

        loss = criterion(out, y)

        with torch.no_grad():
            mae = torch.mean(torch.abs(out - y))
            rmse = torch.sqrt(torch.mean((out - y) ** 2))

            pred = out.detach().cpu().numpy()
            true = y.detach().cpu().numpy()
            all_preds.append(pred)
            all_targets.append(true)

        if mode == "train":
            loss.backward()
            optimizer.step()

        mae_meter.add(mae.item())
        rmse_meter.add(rmse.item())
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            print(f"Step [{i + 1}/{len(data_loader)}], Loss: {loss_meter.value():.4f}, RMSE: {rmse_meter.value():.4f}")

    t_end = time.time()
    total_time = t_end - t_start
    print(f"Epoch time: {total_time:.1f}")
    metrics = {
        f"{mode}_loss": loss_meter.value(),
        f"{mode}_mae": mae_meter.value(),
        f"{mode}_rmse": rmse_meter.value()
        }

    if img_output:
        return metrics, all_preds, all_targets, all_imgs
    else: 
        return metrics, all_preds, all_targets

def prepare_output(config_paths, config):
    os.makedirs(config_paths.res_dir, exist_ok=True)
    if config.fold is None:
        for fold in range(1, 6):
            os.makedirs(os.path.join(config_paths.res_dir, f"Fold_{fold}"), exist_ok=True)
    else:
        os.makedirs(os.path.join(config_paths.res_dir, f"Fold_{config.fold}"), exist_ok=True)

def config_to_json(config):
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

def save_results(fold, metrics, config, config_paths, preds, targets, imgs = None):
    # saving the results

    out_dir = os.path.join(config_paths.res_dir, f"Fold_{fold}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save metrics (MAE, RMSE, R2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 2. Save predictions + ground truth 
    np.save(
        os.path.join(out_dir, "predictions.npy"),
        np.concatenate(preds, axis=0).astype(np.float32)
    )

    np.save(
        os.path.join(out_dir, "targets.npy"),
        np.concatenate(targets, axis=0).astype(np.float32)
    )

    if imgs is not None:
        np.save(
            os.path.join(out_dir, "images.npy"),
            np.concatenate(imgs, axis=0).astype(np.float32)
        )

    # 3. save configs for reproducibility
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_to_json(config), f, indent=4)

    with open(os.path.join(out_dir, "config_paths.json"), "w") as f:
        json.dump(config_to_json(config_paths), f, indent=4)

def overall_performance(all_preds, all_targets,overall=True):
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_true = y_true.reshape(y_true.shape[0], -1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }

    if overall:
        print("\n=== Overall Regression Performance ===")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    return metrics