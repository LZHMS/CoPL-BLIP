from dassl.engine import TRAINER_REGISTRY, TrainerX
import time
from collections import deque
import datetime
import numpy as np
from sklearn.mixture import GaussianMixture
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import ( MetricMeter, AverageMeter, mkdir_if_missing, load_pretrained_weights )
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling.ops.utils import sharpen_prob, create_onehot

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from blip.blip_itm import blip_itm

from datasets.data_manager import DPLDataManager

_tokenizer = _Tokenizer()
from trainers.loss import GeneralizedCrossEntropy

def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        dm = DPLDataManager(self.cfg, custom_tfm_test=preprocess)
        
        self.train_loader_x = dm.train_loader_x
        for batch_idx, batch in enumerate(self.train_loader_x):
            print(batch[0].shape) 
        from IPython import embed
        embed()
        
