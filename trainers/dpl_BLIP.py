import imp
from random import sample
from dassl.engine import TRAINER_REGISTRY, TrainerX
import os.path as osp
import os
import time
import copy
import datetime
import numpy as np
from tqdm import tqdm
import json
import pickle
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from transformers import BertTokenizer
from blip.blip_itm import blip_itm
from functools import partial

from datasets.data_manager import UPLDataManager
from evaluation.evaluator import UPLClassification
from .hhzsclip import ZeroshotCLIP
from .utils import (select_top_k_similarity_per_class, caculate_noise_rate, save_outputs,
select_top_k_similarity, select_top_by_value, caculate_noise_rate_analyze, select_top_k_similarity_per_class_with_noisy_label)

_tokenizer = _Tokenizer()
from trainers.loss import GeneralizedCrossEntropy


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    # semi-supervised templates
    "SSOxfordPets": "a photo of a {}, a type of pet.",
    "SSOxfordFlowers": "a photo of a {}, a type of flower.",
    "SSFGVCAircraft": "a photo of a {}, a type of aircraft.",
    "SSDescribableTextures": "{} texture.",
    "SSEuroSAT": "a centered satellite photo of {}.",
    "SSStanfordCars": "a photo of a {}.",
    "SSFood101": "a photo of {}, a type of food.",
    "SSSUN397": "a photo of a {}.",
    "SSCaltech101": "a photo of a {}.",
    "SSUCF101": "a photo of a person doing {}.",
    "SSImageNet": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())
    
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, features):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = "a photo of"
        classnames = [name.replace("_", " ") for name in classnames]

        ctx_prompts = []
        for i in range(5):
            ctx_prompts.append([ctx_init + " " + name + ", " + features[name][i] + '.' for name in classnames])

        self.n_cls = n_cls
        self.prompts = ctx_prompts


class PromptMatrixLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPL.N_CTX
        n_block = cfg.TRAINER.DPL.N_BLOCK
        ctx_init = cfg.TRAINER.DPL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors = ctx_vectors.expand(n_block, -1, -1)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.DPL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_block, n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_block, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of blocks: {n_block}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :].expand(n_block, -1, -1, -1))  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :].expand(n_block, -1, -1, -1))  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_block = n_block
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DPL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)  # n_cls, n_block, n_ctx, ctx_dim
            ctx = ctx.permute(1, 0, 2, 3)   # n_block, n_cls, n_ctx, ctx_dim

        prefix = self.token_prefix             
        suffix = self.token_suffix             

        if self.class_token_position == "end":
            prompts_matrix = torch.cat(
                [
                    prefix,  # (n_block, n_cls, 1, dim)
                    ctx,     # (n_block, n_cls, n_ctx, dim)
                    suffix,  # (n_block, n_cls, *, dim)
                ],
                dim=2,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_matrix = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i : i + 1, :, :]
                class_i = suffix[:, i : i + 1, :name_len, :]
                suffix_i = suffix[:, i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[:, i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[:, i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (n_block, 1, 1, dim)
                        ctx_i_half1,  # (n_block, 1, n_ctx//2, dim)
                        class_i,      # (n_block, 1, name_len, dim)
                        ctx_i_half2,  # (n_block, 1, n_ctx//2, dim)
                        suffix_i,     # (n_block, 1, *, dim)
                    ],
                    dim=2,
                )
                prompts_matrix.append(prompt)
            prompts_matrix = torch.cat(prompts_matrix, dim=1)

        elif self.class_token_position == "front":
            prompts_matrix = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i : i + 1, :, :]
                class_i = suffix[:, i : i + 1, :name_len, :]
                suffix_i = suffix[:, i : i + 1, name_len:, :]
                ctx_i = ctx[:, i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (n_block, 1, 1, dim)
                        class_i,   # (n_block, 1, name_len, dim)
                        ctx_i,     # (n_block, 1, n_ctx, dim)
                        suffix_i,  # (n_block, 1, *, dim)
                    ],
                    dim=2,
                )
                prompts_matrix.append(prompt)
            prompts_matrix = torch.cat(prompts_matrix, dim=1)

        else:
            raise ValueError

        return prompts_matrix


class CustomBLIP(nn.Module):
    def __init__(self, cfg, classnames, features, device):
        super().__init__()
        self.blip = blip_itm(pretrained=cfg.TRAINER.DPL.BLIP_PATH, image_size=cfg.INPUT.SIZE[0], vit='base')
        self.blip = self.blip.to(device=device)

        self.prompt_learner = PromptLearner(cfg, classnames, features)
        self.device = device
        self.classnames = classnames
        self.cfg = cfg

    def forward(self, image, refurbished_label):
        prompts = self.prompt_learner.prompts

        prediction = []
        for i in range(5):
            refurbished_prompts = [prompts[i][refurbished_label[j].item()] for j in range(len(refurbished_label))]
            itm_output = self.blip(image, refurbished_prompts, match_head='itm')
            itm_score = F.softmax(itm_output, dim=1)[:,1]
            #itc_score = self.blip(image, refurbished_prompts, match_head='itc')
            prediction.append(itm_score)                      
        return torch.stack(prediction, dim=0)
    

class CustomCLIPMatrix(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptMatrixLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip = clip_model
        self.classnames = classnames
        self.cfg = cfg

    def forward(self, image):
        clip_image_features = self.image_encoder(image.type(self.dtype))
        clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        prompts_matrix = self.prompt_learner()  # n_block, n_cls, *, ctx_dim
        #self.ortho_loss = self.ortho_penalty(prompts_matrix)
        tokenized_prompts = self.tokenized_prompts
        logits_matrix = torch.zeros(prompts_matrix.size(0), clip_image_features.size(0), prompts_matrix.size(1))
        texts_matrix = torch.zeros(prompts_matrix.size(0), prompts_matrix.size(1), clip_image_features.size(1))
        for row in range(prompts_matrix.size(0)):
            prompts = prompts_matrix[row, :, :, :]

            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits_matrix[row, :, :] = logit_scale * clip_image_features @ text_features.t()
            texts_matrix[row, :, :] = text_features

        return logits_matrix, clip_image_features, texts_matrix


@TRAINER_REGISTRY.register()
class DPL(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE_loss = GeneralizedCrossEntropy(q=0.5)
        self.gt_label_dict = self.get_gt_label(cfg)
        self.beta = self.cfg.TRAINER.DPL.BETA
        self.clean_samples_num = 0
        self.samples_num = self.dm._num_samples
        self.match_samples = [0, 0, 0]

        # store the analysis results
        self.analysis_results = {'Epoch 1': [], 'Epoch 10': [], 'Epoch 20': [], 'Epoch 30': [], 'Epoch 40': [], 'Epoch 50': []}

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        features = self.dm.dataset.descriptions

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DPL.PREC == "fp32" or cfg.TRAINER.DPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.generator = CustomCLIPMatrix(cfg, classnames, clip_model)     # generator
        self.discriminator = CustomBLIP(cfg, classnames, features, self.device)      # discriminator
        self.n_cls = self.generator.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder")
        print("The params need to be learned in Generator:")
        for name, param in self.generator.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            if param.requires_grad:
                print(name)
        print("The params need to be learned in Discriminator:")
        for name, param in self.discriminator.named_parameters():
            if ("prompt_learner" not in name and "cls_head" not in name) or "ALBEF" in name:
                param.requires_grad_(False)
            if param.requires_grad:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.generator.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.discriminator.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optimG = build_optimizer(self.generator.prompt_learner, cfg.OPTIM)
        self.schedG = build_lr_scheduler(self.optimG, cfg.OPTIM)
        self.register_model("prompt_learner_generator", self.generator.prompt_learner, self.optimG, self.schedG)

        # self.optimD = build_optimizer(self.discriminator.prompt_learner, cfg.OPTIM)
        # self.schedD = build_lr_scheduler(self.optimD, cfg.OPTIM)
        # self.register_model("prompt_learner_discriminator", self.discriminator.prompt_learner, self.optimD, self.schedD)

        self.scaler = GradScaler() if cfg.TRAINER.DPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.generator = nn.DataParallel(self.generator)
            #self.discriminator = nn.DataParallel(self.discriminator)

    def get_gt_label(self, cfg):
            dataset_map = {
                "SSImageNet":"imagenet",
                "SSCaltech101":"caltech-101",
                "SSOxfordPets":"oxford_pets",
                "SSUCF101":"ucf101",
                "SSOxfordFlowers":"oxford_flowers",
                "SSStanfordCars":"stanford_cars",
                "SSFGVCAircraft":"fgvc_aircraft",
                "SSDescribableTextures":"dtd",
                "SSEuroSAT":"eurosat",
                "SSFood101":"food-101",
                "SSSUN397":"sun397"          
            }
            dataset_dir = dataset_map[self.cfg.DATASET.NAME]
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            dataset_dir = os.path.join(root, dataset_dir)
            gt_labels = os.path.join(dataset_dir, "{}_GTlabels.json".format(self.cfg.DATASET.NAME))
            if os.path.exists(gt_labels):
                with open(gt_labels, "rb") as f:
                    gt_label_dict = json.load(f)
                print("Loaded training GT labels from {}".format(gt_labels))
            else:
                print("Generating training GT labels to {}".format(gt_labels))
                gt_label_dict = {}
                for batch_idx, batch in enumerate(self.train_loader_x):
                    input, label, impath = self.parse_batch_test_with_impath(batch)
                    for l, ip in zip(label, impath):
                        ip = './data/' + ip.split('/data/')[1]
                        gt_label_dict[ip] = l.item()
                with open(gt_labels, "w") as outfile:
                    json.dump(gt_label_dict, outfile)
            return gt_label_dict

    def forward_backward(self, batch):
        gt_label_list = []      # ground truth
        image, label, impath = self.parse_batch_test_with_impath(batch)  # sample: image, noisy label, path

        for ip in impath:
            ip = './data/' + ip.split('/data/')[1]
            gt_label = self.gt_label_dict[ip]
            gt_label_list.append(gt_label)
        gt_label = torch.tensor(gt_label_list, dtype=label.dtype).to(label.device)

        prec = self.cfg.TRAINER.DPL.PREC

        # produce the pseudo label using the Generator
        logits_matrix, _, _ = self.generator(image)

        # calculate the average loss for each sample based on prompt matrix
        M, B, C = logits_matrix.shape
        logits_matrix = logits_matrix.permute(1, 0, 2)  # M * B * C => B * M * C
        
        lossG = F.cross_entropy(logits_matrix.reshape(-1, C).to(label.device), label.repeat(1, M).reshape(-1), reduction='none')
        lossG = lossG.view(B, M)
        lossG = lossG.mean(dim=1, keepdim=True)   # B * 1

        # fit a two-component GMM to the prediction loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(lossG.detach().cpu().numpy())
        probs = gmm.predict_proba(lossG.detach().cpu().numpy())
        
        # divide the noisy dataset by generator
        w = probs[:, gmm.means_.argmin()]

        # refurbish the pseudo label
        labels_x = torch.zeros((label.shape[0], self.n_cls), device=label.device).scatter_(1, label.view(-1,1), 1)  # one hot label
        average_logits = torch.mean(logits_matrix, dim=1).squeeze().to(self.device)
        pred_labelG = average_logits.argmax(dim=1)     # generator label
        w_expanded = torch.tensor(w, device=self.device).unsqueeze(1)
        robust_logits = average_logits * (1-w_expanded) + labels_x * w_expanded
        
        all_logits_G = []
        all_labels_G = []
        all_gt_label = []
        # absorb clean samples
        clean_ID = (w_expanded > self.beta).squeeze() & (label == pred_labelG)
        all_logits_G.append(average_logits[clean_ID])
        all_labels_G.append(label[clean_ID])
        all_gt_label.append(gt_label[clean_ID])

        # relabel noisy samples
        noisy_ID = (w_expanded > self.beta).squeeze() & (label != pred_labelG)
        all_logits_G.append(average_logits[noisy_ID])
        all_labels_G.append(robust_logits[noisy_ID].argmax(dim=1))
        all_gt_label.append(gt_label[noisy_ID])

        # rematch undecided samples
        undecided_ID = (w_expanded <= self.beta).squeeze()
        undecided_gt_label = gt_label[undecided_ID]
        # discriminator
        undecided_logits, undecided_image, undecided_label = average_logits[undecided_ID], image[undecided_ID], label[undecided_ID]
        logitsD = self.discriminator(undecided_image, undecided_label)     # P * B
        match_probability = torch.mean(logitsD, dim=0)

        # check the samples
        if match_probability.shape[0] > 1:
            # fit a two-component GMM to the prediction loss
            match_probability = match_probability.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
            gmm.fit(match_probability.detach().cpu().numpy())
            probs = gmm.predict_proba(match_probability.detach().cpu().numpy())
            
            # divide the noisy dataset by generator
            w = probs[:, gmm.means_.argmax()]

            match_ID = w >= 0.5
        else:
            match_ID = match_probability > 0.2
        
        all_logits_G.append(undecided_logits[match_ID])
        all_labels_G.append(undecided_label[match_ID])
        all_gt_label.append(undecided_gt_label[match_ID])

        all_logits_G = torch.cat(all_logits_G)
        all_labels_G = torch.cat(all_labels_G)
        all_gt_label = torch.cat(all_gt_label)
        self.clean_samples_num += all_logits_G.shape[0]

        # save the analysed results
        if self.cfg.TRAINER.DPL.ANALYSIS:
            self.match_samples[0] += sum(undecided_label[match_ID] == undecided_gt_label[match_ID])
            self.match_samples[1] += sum(match_ID)
            self.match_samples[2] += sum(undecided_ID)

        if prec == "amp":
            with autocast():
                loss = F.cross_entropy(all_logits_G, all_labels_G)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.cfg.TRAINER.DPL.USE_ROBUSTLOSS:
                lossG = self.GCE_loss(all_logits_G, all_labels_G)
                #lossD = self.GCE_loss(torch.mean(logitsD, dim=0), labelD)
            else:
                lossG = F.cross_entropy(all_logits_G, all_labels_G)
                #lossD = F.cross_entropy(torch.mean(logitsD, dim=0), labelD)

        self.model_backward_and_update(lossG, "prompt_learner_generator")
        #self.model_backward_and_update(lossD, "prompt_learner_discriminator")

        loss_summary = {
            "lossG": lossG.item(),
            #"lossD": lossD.item(),
            "acc": compute_accuracy(all_logits_G, all_gt_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return logits_matrix, loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]


            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def load_model_by_id(self, directory, model_id, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best-{}.pth.tar'.format(model_id)

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None, trainer_list=None):
        """A generic testing pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME, 'fp'+str(self.cfg.TRAINER.DPL.NUM_FP),
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS)+'_random_init'+str(self.cfg.TRAINER.DPL.CLASS_TOKEN_POSITION))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):
            results_id += 1
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w')
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w')

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        outputs_all = []
        label_all = []
        image_features_all = []
        text_features_all = []
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if trainer_list is None or len(trainer_list)==1:
                logits_matrix, image_features, text_features = self.model_inference(input)
                image_features_all.append(image_features)
                text_features_all.append(text_features)
            
            else:
                # ensemble
                logits_matrix = [t.model_inference(input)[0] for t in trainer_list]
                logits_matrix = sum(logits_matrix) / len(logits_matrix)
            output = torch.mean(logits_matrix, dim=0).squeeze().to(self.device)
            self.evaluator.process(output, label, self.per_image_txt_writer, self.per_class_txt_writer)
            outputs_all.append(output)
            label_all.append(label)
        results = self.evaluator.evaluate()
        if split in ['all', 'train', 'test', 'novel', 'base']:
            if len(outputs_all) != 0:
                outputs_all = torch.cat(outputs_all, dim=0)
                label_all = torch.cat(label_all, dim=0)
                image_features_all = torch.cat(image_features_all, dim=0)
                text_features_all = text_features_all[0]
                torch.save(image_features_all, os.path.join(save_path, '{}_v_features.pt'.format(split)))
                torch.save(image_features_all, os.path.join(save_path, '{}_targets.pt'.format(split)))
                torch.save(outputs_all, os.path.join(save_path, '{}_logits.pt'.format(split)))
                torch.save(text_features_all, os.path.join(save_path, '{}_l_features.pt'.format(split)))


        self.per_image_txt_writer.close()
        self.per_class_txt_writer.close()


        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def zero_shot_analyze(self, trainer_list=None):
        """A generic predicting pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        data_loader = self.train_loader_sstrain
        outputs = []
        image_features_list = []
        labels_list = []
        img_paths = []
        from tqdm import tqdm
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            input, label, impath = self.parse_batch_test_with_impath(batch)
            if trainer_list is None or len(trainer_list)==1:
                output, image_features, text_features = self.model.zero_shot_forward(input, self.device)
            else:
                # ensemble
                outputs = [t.model.zero_shot_forward(input, self.device)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            outputs.append(output)
            image_features_list.append(image_features)
            labels_list.append(label)
            img_paths.append(impath)
        sstrain_outputs = torch.cat(outputs, dim=0)
        sstrain_img_paths = np.concatenate(img_paths, axis=0)
        image_features = torch.cat(image_features_list, axis=0)
        labels_list = torch.cat(labels_list, axis=0)
        # text_features = torch.cat(text_features, axis=0)
        print('labels_list', text_features.shape)
        print('image_features', image_features.shape)
        print('text_features', text_features.shape)
        predict_label_dict, _ = select_top_k_similarity_per_class(sstrain_outputs, sstrain_img_paths, -1, image_features, True)
        save_outputs(self.train_loader_x, self, predict_label_dict, self.cfg.DATASET.NAME, text_features, backbone_name=self.cfg.MODEL.BACKBONE.NAME)
        caculate_noise_rate_analyze(predict_label_dict, train_loader=self.train_loader_x, trainer=self)
        return predict_label_dict


    def load_from_exist_file(self, file_path, model_names):
        logits = None
        for model in model_names:
            model_path = os.path.join(file_path, model)
            logist_path = os.path.join(model_path, '{}_logits.pt'.format(self.cfg.DATASET.NAME))
            if logits is None:
                logits = torch.load(logist_path)
            else:
                logits += torch.load(logist_path)

            info_path = os.path.join(model_path, '{}.json'.format(self.cfg.DATASET.NAME))
            info = json.load(open(info_path))
            items = []
            for c in info:
                for img_path in info[c]:
                    item = info[c][img_path]
                    items.append([img_path, int(item[3])]) # ip, pred_label
            sorted(items, key=(lambda x:x[1]))
            sstrain_img_paths = np.array(items)[:,0]

        logits /= len(model_names)
        predict_label_dict = select_top_k_similarity_per_class_with_noisy_label(img_paths=sstrain_img_paths,
                                                                                K=self.cfg.DATASET.NUM_SHOTS,
                                                                                random_seed=self.cfg.SEED, 
                                                                                gt_label_dict=self.gt_label_dict,
                                                                                num_fp=self.cfg.TRAINER.DPL.NUM_FP)
        return predict_label_dict

    def load_from_no_file(self):
        data_loader = self.train_loader_sstrain
        img_paths = []
        from tqdm import tqdm
        for _, batch in tqdm(enumerate(data_loader)):
            _, _, impath = self.parse_batch_test_with_impath(batch)
            for ip in impath:
                ip = './data/' + ip.split('/data/')[1]
                img_paths.append(ip)
        sstrain_img_paths = np.array(img_paths)
    
        predict_label_dict = select_top_k_similarity_per_class_with_noisy_label(img_paths=sstrain_img_paths,
                                                                                K=self.cfg.DATASET.NUM_SHOTS,
                                                                                random_seed=self.cfg.SEED, 
                                                                                gt_label_dict=self.gt_label_dict,
                                                                                num_fp=self.cfg.TRAINER.DPL.NUM_FP)
        return predict_label_dict
    
    @torch.no_grad()
    def zero_shot_predict(self, trainer_list=None):
        """A generic predicting pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME,
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data_loader = self.train_loader_sstrain

        outputs = []
        img_paths = []


        for batch_idx, batch in tqdm(enumerate(data_loader)):
            input, label, impath = self.parse_batch_test_with_impath(batch)
            if trainer_list is None or len(trainer_list)==1:
                output, image_features, text_features = self.model.zero_shot_forward(input, self.device)
            else:
                # ensemble
                outputs = [t.model.zero_shot_forward(input, self.device)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            outputs.append(output)
            img_paths.append(impath)


        outputs = torch.cat(outputs, dim=0)
        img_paths = np.concatenate(img_paths, axis=0)


        if self.cfg.DATASET.CLASS_EQULE is True:
            if self.cfg.DATASET.CONF_THRESHOLD > 0:
                predict_label_dict_1, predict_conf_dict_1 = select_top_k_similarity_per_class(outputs, img_paths, K=self.cfg.DATASET.NUM_SHOTS)
                predict_label_dict_2, predict_conf_dict_2 = select_top_by_value(outputs, img_paths, conf_threshold=self.cfg.DATASET.CONF_THRESHOLD)

                print(len(predict_label_dict_1), 'predict_label_dict_1')
                print(len(predict_label_dict_2), 'predict_label_dict_2')

                predict_label_dict = dict(predict_label_dict_1, **predict_label_dict_2)
                predict_conf_dict = dict(predict_conf_dict_1, **predict_conf_dict_2)
                caculate_noise_rate(predict_label_dict, train_loader=self.train_loader_x, trainer=self)
                print('select {} samples'.format(len(predict_label_dict)))

            else:
                print("K {} shots".format(self.cfg.DATASET.NUM_SHOTS))
                predict_label_dict, predict_conf_dict = select_top_k_similarity_per_class(outputs, img_paths, K=self.cfg.DATASET.NUM_SHOTS)
                caculate_noise_rate(predict_label_dict,  train_loader=self.train_loader_x, trainer=self)
                print('select {} samples'.format(len(predict_label_dict)))

        else:
            print("K", self.cfg.DATASET.NUM_SHOTS*text_features.shape[0])
            predict_label_dict, predict_conf_dict = select_top_k_similarity(outputs, img_paths, K=self.cfg.DATASET.NUM_SHOTS*text_features.shape[0])
            caculate_noise_rate(predict_label_dict, train_loader=self.train_loader_x, trainer=self)
            print('select {} samples'.format(len(predict_label_dict)))
        return predict_label_dict, predict_conf_dict

    @torch.no_grad()
    def zero_shot_test(self, split=None, trainer_list=None):
        """A generic predicting pipeline."""

        self.set_model_mode("eval")
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME,
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):
            results_id += 1
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w')
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w')

        if split is None:
            split = self.cfg.TEST.SPLIT

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME,
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        elif split=="train":
            data_loader = self.train_loader_x
            print("Do evaluation on train set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label, impath = self.parse_batch_test_with_impath(batch)
            if trainer_list is None or len(trainer_list)==1:
                output, image_features, text_features = self.model.zero_shot_forward(input, self.device)
            else:
                # ensemble
                outputs = [t.model.zero_shot_forward(input, self.device)[0] for t in trainer_list]
                output = sum(outputs) / len(outputs)
            self.evaluator.process(output, label, self.per_image_txt_writer, self.per_class_txt_writer)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        self.per_image_txt_writer.close()
        self.per_class_txt_writer.close()

        return list(results.values())[0]



    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        dm = UPLDataManager(self.cfg, custom_tfm_test=preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.train_loader_sstrain = dm.train_loader_sstrain
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        if self.cfg.DATALOADER.OPEN_SETTING:
            self.test_novel_loader = dm.test_novel_loader
            self.test_base_loader = dm.test_base_loader


        self.dm = dm

    def sstrain_with_id(self, model_id):
        self.sstrain(self.start_epoch, self.max_epoch, model_id)

    def sstrain(self, start_epoch, max_epoch, model_id):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch_with_sstrain()
            self.after_epoch(model_id)

        self.after_train(model_id)

    def run_epoch_with_sstrain(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_sstrain)

        end = time.time()
        self.clean_samples_num = 0
        self.match_samples = [0, 0, 0]
        print(f"The threadhold for selecting samples: {self.beta}")
        for self.batch_idx, batch in enumerate(self.train_loader_sstrain):
            data_time.update(time.time() - end)
            _, loss_summary = self.forward_backward(batch)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(),
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
            
    def after_epoch(self, model_id):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        print(f"clean samples: {self.clean_samples_num}, total samples: {self.samples_num}")
        self.beta = self.cfg.TRAINER.DPL.BETA * (1 + self.clean_samples_num/self.samples_num)

        acc_rate = self.match_samples[0] / self.match_samples[1]
        match_rate = self.match_samples[1] / self.match_samples[2]
        print(f"acc rate: {acc_rate}, match rate: {match_rate}, match samples: {self.match_samples[0]} / {self.match_samples[1]} / {self.match_samples[2]}")
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best-{}.pth.tar".format(model_id)
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(
                    self.epoch,
                    self.output_dir,
                    model_name="model-best-{}.pth.tar".format(model_id)
                )

    def after_train(self, model_id):
        print("Finished training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model_by_id(self.output_dir, model_id)

            self.test(split='test')

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # save the results
        save_path = os.path.join(self.output_dir, 'analysis.json')
        with open(save_path, 'w') as json_file:
            json.dump(self.analysis_results, json_file, indent=4)

        # Close writer
        self.close_writer()

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def parse_batch_test_with_impath(self, batch):
        input = batch["img"]
        label = batch["label"]
        impath = batch["impath"]

        input = input.to(self.device)

        label = label.to(self.device)

        return input, label, impath

    @torch.no_grad()
    def test_with_existing_logits(self, logits, split='test'):

        self.set_model_mode("eval")
        self.evaluator.reset()

        save_path = os.path.join(self.cfg.TEST.Analyze_Result_Path, self.cfg.DATASET.NAME, 'fp'+str(self.cfg.TRAINER.DPL.NUM_FP),
        str(self.cfg.OPTIM.MAX_EPOCH)+'_'+str(self.cfg.SEED)+'_'+str(self.cfg.DATASET.NUM_SHOTS)+'_random_init'+str(self.cfg.TRAINER.DPL.CLASS_TOKEN_POSITION))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_id = 0
        while os.path.exists(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id))):
            results_id += 1
        self.per_image_txt_writer = open(os.path.join(save_path, 'per_image_results_{}_{}.txt'.format(split, results_id)), 'w')
        self.per_class_txt_writer = open(os.path.join(save_path, 'per_class_results_{}_{}.txt'.format(split, results_id)), 'w')

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        elif split=="novel":
            data_loader = self.test_novel_loader
            print("Do evaluation on test novel set")
        elif split=="base":
            data_loader = self.test_base_loader
            print("Do evaluation on test base set")
        elif split=="all":
            data_loader = self.test_loader
            print("Do evaluation on test set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        label_all = []
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            label_all.append(label)
        label_all = torch.hstack(label_all)
        print(label_all.shape)

        self.evaluator.process(logits, label_all, self.per_image_txt_writer, self.per_class_txt_writer)
        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results