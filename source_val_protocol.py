"""Historical Full-MLUDA Houston checkpoint selection and evaluation.

Core split/patch/evaluation functions are copied verbatim from recovered commit
4b467cba6950d6f964e9c515031fbabed651d388, train_full_mluda_scene_shift.py.
The wrapper isolates evaluation RNG and restores all module training modes.
"""
import math
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch import nn
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

CLASS_NUM = 7
BATCH_SIZE = 32

def center_patches(cube: np.ndarray, centers: np.ndarray, width: int = 7) -> np.ndarray:
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    out = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for index, (row, col) in enumerate(centers):
        out[index] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return out

def source_split(gt: np.ndarray, split_seed: int):
    """Reproduce the official 180-per-class split with a local RNG."""
    rng = np.random.RandomState(split_seed)
    centers_train, centers_val = [], []
    for class_id in range(1, CLASS_NUM + 1):
        centers = np.argwhere(gt == class_id)
        rng.shuffle(centers)
        centers_train.append(centers[:180])
        centers_val.append(centers[180:])
    train = np.concatenate(centers_train).astype(np.int64)
    val = np.concatenate(centers_val).astype(np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    train_y = gt[train[:, 0], train[:, 1]].astype(np.int64) - 1
    val_y = gt[val[:, 0], val[:, 1]].astype(np.int64) - 1
    return train, train_y, val, val_y

def evaluate_source(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, x)[3]
            loss_sum += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return loss_sum / count, correct / count

def evaluate_target(model, target_cube, target_gt, source_reference, device):
    centers = np.argwhere(target_gt > 0)
    labels = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(centers), 512):
            patches = torch.from_numpy(center_patches(target_cube, centers[start:start + 512])).to(device)
            for offset in range(0, len(patches), BATCH_SIZE):
                target_batch = patches[offset:offset + BATCH_SIZE]
                repeats = math.ceil(len(target_batch) / len(source_reference))
                reference = source_reference.repeat((repeats, 1, 1, 1))[:len(target_batch)].to(device)
                predictions.append(model(reference, target_batch)[8].argmax(1).cpu().numpy())
    prediction = np.concatenate(predictions)
    confusion = metrics.confusion_matrix(labels, prediction, labels=np.arange(CLASS_NUM))
    per_class = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    return {
        "oa": float((prediction == labels).mean()),
        "aa": float(per_class.mean()),
        "kappa": float(metrics.cohen_kappa_score(labels, prediction, labels=np.arange(CLASS_NUM))),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": np.bincount(prediction, minlength=CLASS_NUM).tolist(),
        "num_target_evaluation_samples": int(len(labels)),
    }

@contextmanager
def isolated_evaluation(model):
    modes = [(module, module.training) for module in model.modules()]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            with torch.no_grad():
                yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for module, mode in modes:
            module.training = mode


def validation_data(cube, gt, seed, actual_train_x, actual_train_y):
    train, train_y, val, val_y = source_split(gt, seed)
    train_x = center_patches(cube, train)
    if not (np.array_equal(train_x, actual_train_x)
            and np.array_equal(train_y, actual_train_y)):
        raise ValueError("Historical source split does not match actual training data")
    train_ids = np.ravel_multi_index(train.T, gt.shape)
    val_ids = np.ravel_multi_index(val.T, gt.shape)
    assert len(np.intersect1d(train_ids, val_ids)) == 0
    assert len(train_ids) + len(val_ids) == int((gt > 0).sum())
    loader = DataLoader(TensorDataset(torch.from_numpy(center_patches(cube, val)),
                                     torch.from_numpy(val_y)),
                        batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                        generator=torch.Generator().manual_seed(seed))
    return loader, torch.from_numpy(train_x[:BATCH_SIZE]), {
        "train_centers": train, "train_labels": train_y,
        "validation_centers": val, "validation_labels": val_y,
    }


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def improves_source_validation(row, previous):
    # Historical strict-greater comparison keeps the earliest exact tie.
    return previous is None or row["source_val_accuracy"] > previous["source_val_accuracy"]
