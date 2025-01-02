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

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from blip.blip_itm import blip_itm

from datasets.data_manager import DPLDataManager
from .utils import (select_top_k_similarity_per_class, caculate_noise_rate, save_outputs,
select_top_k_similarity, select_top_by_value, caculate_noise_rate_analyze, select_top_k_similarity_per_class_with_noisy_label)

_tokenizer = _Tokenizer()
from trainers.loss import GeneralizedCrossEntropy


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

def load_blip_to_cpu(cfg):
    pretrained = cfg.TRAINER.DPL.BLIP_PATH
    img_size = cfg.INPUT.SIZE[0]
    blip = blip_itm(pretrained=pretrained, image_size=img_size, vit='base')
    blip = blip.to(device="cpu")
    
    return blip


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
    def __init__(self, cfg, classnames, blip_model, features):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = cfg.TRAINER.DPL.CTX_INIT
        n_ctx = len(ctx_init.split(" "))
        tokenizer = blip_model.tokenizer
        embeddings = blip_model.text_encoder.embeddings
        
        classnames = [name.replace("_", " ") for name in classnames]
        feature_lens = [len(_tokenizer.encode(features[name][0])) for name in classnames]
        ctx_prompts = [ctx_init + " " + name + ", " + features[name][0] + '.' for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        self.token_sos = []
        self.token_cls = []
        self.token_eos = []
        ctx_vectors_prefix_list = []
        ctx_vectors_suffix_list = []
        attention_mask = []
        for i, prompt in enumerate(ctx_prompts):
            prompt = prompt.replace("_", " ")	
            ctx_prompt = tokenizer(prompt, padding='max_length', truncation=True, max_length=35, 
                                return_tensors="pt")
            attention_mask.append(ctx_prompt.attention_mask)
            with torch.no_grad():
                ctx_embedding = embeddings(input_ids=ctx_prompt.input_ids)
        
            self.token_sos.append(ctx_embedding[0, : 1, :])
            ctx_vectors_prefix_list.append(ctx_embedding[0, 1 : 1 + n_ctx, :])
            self.token_cls.append(ctx_embedding[0, 1 + n_ctx : 1 + n_ctx + name_lens[i], :])
            ctx_vectors_suffix_list.append(ctx_embedding[0, 1 + n_ctx + name_lens[i] : 1 + n_ctx + name_lens[i] + feature_lens[i], :])
            self.token_eos.append(ctx_embedding[0, 1 + n_ctx + name_lens[i] + feature_lens[i]:, :])
        
        self.attention_mask = torch.stack(attention_mask, dim=0)
        ctx_vectors_prefix = torch.cat(ctx_vectors_prefix_list, dim=0)
        ctx_vectors_suffix = torch.cat(ctx_vectors_suffix_list, dim=0)

        print(f'Initial context: "{ctx_init}"')
        print(f"Max Number of context words (tokens): {max(feature_lens)}")

        self.ctx_prefix = nn.Parameter(ctx_vectors_prefix)  # to be optimized
        self.ctx_suffix = nn.Parameter(ctx_vectors_suffix)  # to be optimized
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.feature_lens = feature_lens
        self.class_token_position = cfg.TRAINER.DPL.CLASS_TOKEN_POSITION
    
    def forward(self):
        ctx_prefix = self.ctx_prefix
        ctx_suffix = self.ctx_suffix
        
        prompts = []
        last_index = 0
        for i in range(self.n_cls):
            feature_len = self.feature_lens[i]
            token_sos_i = self.token_sos[i]
            token_cls_i = self.token_cls[i]
            token_eos_i = self.token_eos[i]

            ctx_prefix_i = ctx_prefix[i*self.n_ctx : (i+1)*self.n_ctx, :]
            ctx_suffix_i = ctx_suffix[last_index : last_index + feature_len]
            prompts.append(torch.cat(
                [
                    token_sos_i.to(ctx_prefix_i.device),   # (1, dim)
                    ctx_prefix_i,                          # (n_ctx, dim)
                    token_cls_i.to(ctx_prefix_i.device),   # (n_cls, dim)
                    ctx_suffix_i,                          # (n_feature, dim)
                    token_eos_i.to(ctx_prefix_i.device),   # (*, dim)
                ],
                dim=0,
            ))
            last_index += feature_len

        prompts = torch.stack(prompts, dim=0) # (n_cls, n_ctx, dim)
        return prompts

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
    def __init__(self, cfg, classnames, blip_model, features):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, blip_model, features)
        self.attention_mask = self.prompt_learner.attention_mask
        self.image_encoder = blip_model.visual_encoder
        self.text_encoder = blip_model.text_encoder
        self.itm_head = blip_model.itm_head
        self.vision_proj = blip_model.vision_proj
        self.text_proj = blip_model.text_proj

        self.blip = blip_model
        self.classnames = classnames
        self.cfg = cfg

    def forward(self, image, refurbished_label=None, match_head='itm'):
        image_embeds = self.image_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        prompts = self.prompt_learner()  # (n_cls, n_ctx, dim)
        if match_head == 'itm':
            refurbished_prompts = prompts[refurbished_label, :, :]
            output = self.text_encoder(encoder_embeds = refurbished_prompts,
                                attention_mask = self.attention_mask[refurbished_label, :].to(refurbished_prompts.device),
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,        
                                return_dict = True,
                                )
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])           
            return itm_output    # Samples_Num * 2
       
        elif match_head == 'itc':
            text_output = self.text_encoder(encoder_embeds = prompts,
                                attention_mask = self.attention_mask.to(prompts.device),                    
                                return_dict = True, 
                                mode = 'text')                     
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    
            sim = image_feat @ text_feat.t()        
            return sim
    

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
        self.beta = self.cfg.TRAINER.DPL.BETA
        self.clean_samples_num = 0
        self.samples_num = self.dm._num_samples

        # store the analysis results
        self.analysis_results = {'Epoch 1': [], 'Epoch 10': [], 'Epoch 20': [], 'Epoch 30': [], 'Epoch 40': [], 'Epoch 50': []}

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPL.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (except self.dm).
        """
        _, preprocess = clip.load(self.cfg.MODEL.BACKBONE.NAME)
        dm = DPLDataManager(self.cfg, custom_tfm_test=preprocess)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        features = self.dm.dataset.features

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        blip_model = load_blip_to_cpu(cfg)

        if cfg.TRAINER.DPL.PREC == "fp32" or cfg.TRAINER.DPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            blip_model.float()

        print("Building custom CLIP")
        self.generator = CustomCLIPMatrix(cfg, classnames, clip_model)     # generator
        print("Building custom BLIP")
        self.discriminator = CustomBLIP(cfg, classnames, blip_model, features)      # discriminator
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
            if "prompt_learner" not in name:
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

        self.optimD = build_optimizer(self.discriminator.prompt_learner, cfg.OPTIM)
        self.schedD = build_lr_scheduler(self.optimD, cfg.OPTIM)
        self.register_model("prompt_learner_discriminator", self.discriminator.prompt_learner, self.optimD, self.schedD)

        self.scaler = GradScaler() if cfg.TRAINER.DPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        self.clean_samples_num = 0
        print(f"The threadhold for selecting samples: {self.beta}")
        for self.batch_idx, batch in enumerate(self.train_loader_x):
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
        
        print(f"clean samples: {self.clean_samples_num}, total samples: {self.samples_num}")
        self.beta = self.cfg.TRAINER.DPL.BETA * (1 + self.clean_samples_num/self.samples_num)

    def forward_backward(self, batch):
        image, label = self.parse_batch_test(batch)  # sample: image, noisy label, path

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
        similarity = F.softmax(average_logits, dim=1).max(dim=1)[0]
        pred_labelG = average_logits.argmax(dim=1)     # generator label
        w_expanded = torch.tensor(w, device=self.device).unsqueeze(1)
        robust_logits = average_logits * (1-w_expanded) + labels_x * w_expanded
        
        all_logits_G = []
        all_labels_G = []
        all_images_G = []
        # absorb clean samples
        clean_ID = (similarity > self.beta).squeeze() & (label == pred_labelG)
        all_logits_G.append(average_logits[clean_ID])
        all_labels_G.append(label[clean_ID])
        all_images_G.append(image[clean_ID])

        # relabel noisy samples
        noisy_ID = (similarity > self.beta).squeeze() & (label != pred_labelG)
        all_logits_G.append(average_logits[noisy_ID])
        all_labels_G.append(robust_logits[noisy_ID].argmax(dim=1))
        all_images_G.append(image[noisy_ID])

        # rematch undecided samples
        undecided_ID = (similarity <= self.beta).squeeze()
        if sum(undecided_ID) > 0:
            # discriminator
            undecided_logits, undecided_image, undecided_label = average_logits[undecided_ID], image[undecided_ID], label[undecided_ID]
            labelD = torch.tensor(w_expanded[undecided_ID] >= 0.5, dtype=torch.long).squeeze(dim=1) #torch.tensor(undecided_gt_label == undecided_label, dtype=torch.long)  # Prompts_Num * Samples_Num * 2
            logitsD = self.discriminator(undecided_image, undecided_label, 'itm')        # Samples_Num * 2
            match_probability = logitsD[:, 1]
            sim = self.discriminator(undecided_image, match_head='itc')        # Samples_Num * n_cls
            sim = F.softmax(sim, dim=1)
            pred_labelU, pi_u =  sim.argmax(dim=1), sim.max(dim=1)[0]

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

            # rematch the label
            labels_u = undecided_label.clone()
            labels_temp = torch.zeros((labels_u.shape[0], self.n_cls), device=labels_u.device).scatter_(1, labels_u.view(-1,1), 1)  # one hot label
            robust_logits = sim * pi_u.unsqueeze(1) + labels_temp * (1-pi_u.unsqueeze(1))
            labels_u[match_ID == 1] = robust_logits[match_ID == 1, :].argmax(dim=1)
            labels_u[match_ID == 0] = pred_labelU[match_ID == 0]
            

            all_logits_G.append(undecided_logits)
            all_labels_G.append(labels_u)
            all_images_G.append(undecided_image)

        all_logits_G = torch.cat(all_logits_G)
        all_labels_G = torch.cat(all_labels_G)
        all_images_G = torch.cat(all_images_G)

        # evaluate the quality
        logitsAll = self.discriminator(all_images_G, all_labels_G, 'itm')        # Samples_Num * 2
        match_probability = logitsAll[:, 1]
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

        self.clean_samples_num += (all_logits_G.shape[0] - sum(undecided_ID))

        if prec == "amp":
            with autocast():
                loss = F.cross_entropy(all_logits_G, all_labels_G)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.cfg.TRAINER.DPL.USE_ROBUSTLOSS:
                lossG = self.GCE_loss(all_logits_G[match_ID, :], all_labels_G[match_ID])
                if sum(undecided_ID) > 0:
                    lossD = self.GCE_loss(logitsD, labelD) #+ self.GCE_loss(sim, undecided_gt_label)
                    self.model_backward_and_update(lossD, "prompt_learner_discriminator")
            else:
                lossG = F.cross_entropy(all_logits_G[match_ID, :], all_labels_G[match_ID])
                if sum(undecided_ID) > 0:
                    lossD = F.cross_entropy(logitsD, labelD) #+ F.cross_entropy(sim, undecided_gt_label)
                    self.model_backward_and_update(lossD, "prompt_learner_discriminator")
        self.model_backward_and_update(lossG, "prompt_learner_generator")
        
        if sum(undecided_ID) > 0:
            loss_summary = {
                "lossG": lossG.item(),
                "lossD": lossD.item(),
                "acc": compute_accuracy(all_logits_G, all_labels_G)[0].item(),
            }
        else:
            loss_summary = {
                "lossG": lossG.item(),
                "acc": compute_accuracy(all_logits_G, all_labels_G)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

