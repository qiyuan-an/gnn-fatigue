import sys
import time

import numpy as np
import pandas
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from common import get_model_resgcn
from utils import AverageMeter
from datasets import dataset_factory
from datasets.augmentation import ShuffleSequence, SelectSequenceCenter, ToTensor, MultiInput
from datasets.graph import Graph


def evaluate_fatigue(data_loader, model, log_interval=10, use_flip=False):
    """
    done: write custom evaluate fn
    """
    model.eval()
    batch_time = AverageMeter()

    # Calculate embeddings
    with torch.no_grad():
        end = time.time()
        embeddings = dict()
        for idx, (points, target) in enumerate(data_loader):
            if isinstance(points, list):
                points = points[0]
            if use_flip:
                bsz = points.shape[0]
                data_flipped = torch.flip(points, dims=[1])
                points = torch.cat([points, data_flipped], dim=0)

            if torch.cuda.is_available():
                points = points.cuda(non_blocking=True)

            preds = model(points)

            preds_np = preds.detach().cpu().numpy()
            targets_np = target.detach().cpu().numpy()

            
            # Calculate metrics
            p = precision_score(targets_np, preds_np, average='binary')
            r = recall_score(targets_np, preds_np, average='binary')
            f1 = f1_score(targets_np, preds_np, average='binary')
            acc = accuracy_score(targets_np, preds_np)

    return p, r, f1, acc
