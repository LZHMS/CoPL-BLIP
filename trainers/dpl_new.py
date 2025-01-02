from dassl.engine import TRAINER_REGISTRY, TrainerX
import time
from collections import deque
import datetime
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

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
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPL.N_CTX
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
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.DPL.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

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
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DPL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class CustomBLIP(nn.Module):
    def __init__(self, cfg, classnames, blip_model, features):
        super().__init__()
        ctx_init = "a photo of"
        classnames = [name.replace("_", " ") for name in classnames]
        self.blip = blip_model
        
        self.prompts = [ctx_init + " " + name + ", " + features[name] + '.' for name in classnames]
        self.classnames = classnames
        self.cfg = cfg

    def forward(self, image, refurbished_label):
        prompts = self.prompts

        refurbished_prompts = [prompts[refurbished_label[j].item()] for j in range(len(refurbished_label))]
        itm_output = self.blip(image, refurbished_prompts, match_head='itm')
        itm_score = F.softmax(itm_output, dim=1)[:,1]                    
        return itm_score
    
def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1), reduction='none')

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

@TRAINER_REGISTRY.register()
class DPL(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE = GeneralizedCrossEntropy(q=0.5)
        self.warmup_epoch = cfg.TRAINER.DPL.WARMUP_EPOCH
        self.temp = cfg.TRAINER.DPL.TEMP
        self.beta = cfg.TRAINER.DPL.BETA
        self.alpha1 = cfg.TRAINER.DPL.ALPHA1
        self.alpha2 = cfg.TRAINER.DPL.ALPHA2
        self.theta = 0.01
        self.co_lambda = cfg.TRAINER.DPL.CO_LAMBDA

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
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.fmodel = CustomCLIP(cfg, classnames, clip_model)
        self.blip = CustomBLIP(cfg, classnames, blip_model, features)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        for name, param in self.fmodel.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        for name, param in self.blip.named_parameters():
            param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.fmodel.to(self.device)
        self.blip.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner_A", self.model.prompt_learner, self.optim, self.sched)

        self.foptim = build_optimizer(self.fmodel.prompt_learner, cfg.OPTIM)
        self.fsched = build_lr_scheduler(self.foptim, cfg.OPTIM)
        self.register_model("prompt_learner_B", self.fmodel.prompt_learner, self.foptim, self.fsched)

        self.scaler = GradScaler() if cfg.TRAINER.DPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.fmodel = nn.DataParallel(self.fmodel)
            self.blip = nn.DataParallel(self.blip)

    def train(self):
        """Generic training loops."""

        print("Start WarmUp")
        for self.epoch in range(0, self.warmup_epoch):
            self.warmup()

        self.before_train()
        for self.epoch in range(self.start_epoch + self.warmup_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        
        self.after_train()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # co-divide
        if (self.epoch - self.warmup_epoch) % 5 == 0:
            self.match_scores_A, self.match_ID_A, self.refined_labels_A, self.refined_labels_expand_A = self.eval_train(self.model)
            self.match_scores_B, self.match_ID_B, self.refined_labels_B, self.refined_labels_expand_B = self.eval_train(self.fmodel)

        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def warmup(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_warmup(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def eval_train(self, model):
        self.set_model_mode("eval")
        
        data_len = len(self.train_loader_x.dataset)
        #--- Step 1: do eval for splitting the dataset
        losses = torch.zeros(data_len)     # for GMM modeling
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, _, _ = self.parse_batch(batch)
                output = 0
                for input_i in input:
                    output_i = model(input_i)
                    output += output_i
                output /= len(input)
                probs = torch.softmax(output, dim=1)

                loss = F.cross_entropy(output, label, reduction='none')
                regular = -torch.sum(probs.log() * probs, dim=1)
                loss = loss + regular
                for b in range(label.size(0)):
                    losses[index[b]] = loss[b]

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_loss)
        probs = gmm.predict_proba(input_loss)
        mean = gmm.means_.reshape(-1)
        std = np.sqrt(gmm.covariances_).reshape(-1)
        idx_clean = mean.argmin()
        idx_noise = mean.argmax()

        mean_clean = torch.tensor(mean[idx_clean]).cuda()
        mean_noise = torch.tensor(mean[idx_noise]).cuda()
        std_clean = torch.tensor(std[idx_clean]).cuda()
        std_noise = torch.tensor(std[idx_noise]).cuda()

        # calculate the thredhold
        alpha_1 = mean_clean + torch.sqrt(-2 * (std_clean ** 2) * torch.log(self.theta * std_clean * torch.sqrt(torch.tensor(2 * torch.pi)) + 1e-8))
        alpha_2 = mean_noise - torch.sqrt(-2 * (std_noise ** 2) * torch.log(self.theta * std_noise * torch.sqrt(torch.tensor(2 * torch.pi)) + 1e-8))
	
        if alpha_1 > alpha_2:
            clean_ID = (input_loss < alpha_2.item())
            noisy_ID = (input_loss > alpha_1.item())
        else:
            clean_ID = (input_loss < alpha_1.item())
            noisy_ID = (input_loss > alpha_2.item())
        confused_ID = ~(clean_ID | noisy_ID)     # confusing samples

        # clean probalities for the label
        clean_probs = torch.tensor(probs[:, idx_clean], device=self.device).reshape(-1, 1)

        clean_ID = torch.nonzero(clean_ID, as_tuple=True)[0]
        noisy_ID = torch.nonzero(noisy_ID, as_tuple=True)[0]
        confused_ID = torch.nonzero(confused_ID, as_tuple=True)[0]
        
        #--- Step 2: do label refinement for the three subsets
        noisy_labels = torch.zeros(data_len, dtype=torch.long)
        gt_labels = torch.zeros(data_len, dtype=torch.long)
        refined_labels = torch.zeros(data_len, dtype=torch.long)
        refined_labels_expand = torch.zeros((data_len, self.num_classes))
        itm_scores = torch.zeros(data_len)
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, label_onehot, gt_label = self.parse_batch(batch)

                output = 0
                for input_i in input:
                    output_i = model(input_i)
                    output += output_i     
                output /= len(input)
                probs = torch.softmax(output, dim=1)

                batch_clean_probs = clean_probs[index]

                # label refinement
                refined_label = sharpen_prob(probs, self.temp)

                # label mixrefinement
                mixrefined_predict = batch_clean_probs * label_onehot + (1 - batch_clean_probs) * probs 
                mixrefined_label = sharpen_prob(mixrefined_predict, self.temp)

                batch_refined_labels = label.detach().clone()
                for i, id in enumerate(index):
                    if id in clean_ID:
                        # Label absorb of labeled samples
                        refined_labels[id] = label[i]
                        refined_labels_expand[id] = label_onehot[i]
                        batch_refined_labels[i] = label[i]
                    elif id in noisy_ID:
                        # Label refinement for unlabeled data
                        refined_labels[id] = refined_label[i].argmax()
                        refined_labels_expand[id] = refined_label[i]
                        batch_refined_labels[i] = refined_labels[id]
                    else:
                        # mixrefine confused samples
                        refined_labels[id] = mixrefined_label[i].argmax()
                        refined_labels_expand[id] = mixrefined_label[i]
                        batch_refined_labels[i] = refined_labels[id]
                    noisy_labels[id] = label[i]
                    gt_labels[id] = gt_label[i]

                #--- Step 3: do pesudo label evaluation
                # discriminator
                with torch.no_grad():
                    itm_score = 0
                    for input_i in input:
                        itm_score += self.blip(input_i, batch_refined_labels)
                    itm_score /= len(input)
                    for b in range(label.size(0)):
                        itm_scores[index[b]] = itm_score[b]
        
        itm_scores = (itm_scores - itm_scores.min()) / (itm_scores.max() - itm_scores.min())
        input_match_prob = itm_scores.reshape(-1, 1)

        # fit a two-component GMM to the match probality
        input_match_prob = input_match_prob.cpu()
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_match_prob)
        probs = gmm.predict_proba(input_match_prob)

        # divide the pseudo labels into match and unmatch set
        match_scores = probs[:, gmm.means_.argmax()]
        match_ID = torch.tensor(match_scores >= 0.5)

        # effectiveness for nosiy label refinement
        noisy_rate = sum(noisy_labels != gt_labels) / data_len
        refined_noisy_rate = sum(refined_labels != gt_labels) / data_len
        matched_refined_noisy_rate = sum(refined_labels[match_ID] != gt_labels[match_ID]) / sum(match_ID)
        unmatched_refined_noisy_rate = sum(refined_labels[~match_ID] != gt_labels[~match_ID]) / sum(~match_ID)
        print(f">>> noisy rate: {noisy_rate:.2f} --> refined noisy rate: {refined_noisy_rate:.2f} --> matched refined noisy rate: {matched_refined_noisy_rate:.2f} & unmatched refined noisy rate: {unmatched_refined_noisy_rate:.2f} <<<")

        match_ID = torch.nonzero(match_ID, as_tuple=True)[0]
        return match_scores, match_ID, refined_labels, refined_labels_expand
    

    def forward_backward(self, batch):
        input, label, index, _, _, _ = self.parse_batch(batch)
        negloss = NegEntropy()
        index = [index] * len(input)
        index = torch.cat(index, 0)
        input = torch.cat(input, 0)

        input_x_A, label_x_A, input_x_B, label_x_B = [], [], [], []
        input_u, label_u_A, label_u_B, match_scores_u_A, match_scores_u_B = [], [], [], [], []
        for i, id in enumerate(index):
            if id.item() in self.match_ID_A:
                input_x_A.append(input[i])
                label_x_A.append(self.refined_labels_expand_A[id.item()])

            if id.item() in self.match_ID_B:
                input_x_B.append(input[i])
                label_x_B.append(self.refined_labels_expand_B[id.item()])
            
            if id.item() not in self.match_ID_A and id.item() not in self.match_ID_B:
                input_u.append(input[i])
                match_scores_u_A.append(self.match_scores_A[id.item()])
                match_scores_u_B.append(self.match_scores_B[id.item()])
                label_u_A.append(self.refined_labels_expand_A[id.item()])
                label_u_B.append(self.refined_labels_expand_B[id.item()])

        input_x_A = torch.stack(input_x_A, dim=0).to(self.device)
        input_x_B = torch.stack(input_x_B, dim=0).to(self.device)
        label_x_A = torch.stack(label_x_A, dim=0).to(self.device)
        label_x_B = torch.stack(label_x_B, dim=0).to(self.device)
        match_scores_u_A = torch.tensor(match_scores_u_A).reshape(-1, 1).to(self.device)
        match_scores_u_B = torch.tensor(match_scores_u_B).reshape(-1, 1).to(self.device)
        input_u = torch.stack(input_u, dim=0).to(self.device)
        label_u_A = torch.stack(label_u_A, dim=0).to(self.device)
        label_u_B = torch.stack(label_u_B, dim=0).to(self.device)

        with torch.no_grad():
            output_u = match_scores_u_A * label_u_A + (1-match_scores_u_A) * F.softmax(self.model(input_u), dim=1) + match_scores_u_B * label_u_B + (1-match_scores_u_B) * F.softmax(self.fmodel(input_u), dim=1)
            label_u = sharpen_prob(output_u, self.temp)

        all_inputs_A = torch.cat([input_x_B, input_u], dim=0).to(self.device)
        all_labels_A = torch.cat([label_x_B, label_u], dim=0).to(self.device)

        all_inputs_B = torch.cat([input_x_A, input_u], dim=0).to(self.device)
        all_labels_B = torch.cat([label_x_A, label_u], dim=0).to(self.device)
        
        output_A = self.model(all_inputs_A)
        output_B = self.fmodel(all_inputs_B)

        penalty_A = negloss(output_A)
        penalty_B = negloss(output_B)

        loss_A = F.cross_entropy(output_A, all_labels_A) #+ self.alpha1 * penalty_A
        loss_B = F.cross_entropy(output_B, all_labels_B) #+ self.alpha1 * penalty_B
        self.model_backward_and_update(loss_A, "prompt_learner_A")
        self.model_backward_and_update(loss_B, "prompt_learner_B")

        loss_summary = {
            "loss A": loss_A.item(),
            "loss B": loss_B.item(),
            "acc A": compute_accuracy(output_A, all_labels_A.argmax(dim=1))[0].item(),
            "acc B": compute_accuracy(output_B, all_labels_B.argmax(dim=1))[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def forward_backward_warmup(self, batch):
        input, label, index, _, _, _ = self.parse_batch(batch)
        negloss = NegEntropy()
        label = [label] * len(input)
        label = torch.cat(label, 0)
        input = torch.cat(input, 0)

        prec = self.cfg.TRAINER.DPL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(input)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_A = self.model(input)
            output_B = self.fmodel(input)
            loss_A = F.cross_entropy(output_A, label)
            loss_B = F.cross_entropy(output_B, label)

            penalty_A = negloss(output_A)
            penalty_B = negloss(output_B)
            loss_A += self.alpha1 * penalty_A
            loss_B += self.alpha1 * penalty_B
            self.model_backward_and_update(loss_A, "prompt_learner_A")
            self.model_backward_and_update(loss_B, "prompt_learner_B")

        loss_summary = {
            "loss A": loss_A.item(),
            "loss B": loss_B.item(),
            "acc A": compute_accuracy(output_A, label)[0].item(),
            "acc B": compute_accuracy(output_B, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference_A(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        self.evaluator.reset()
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference_B(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference_A(self, input):
        return self.model(input)
    
    def model_inference_B(self, input):
        return self.fmodel(input)
    
    def parse_batch(self, batch):
        input = []
        for k in range(self.cfg.DATALOADER.K):
            keyname = "img"
            if (k + 1) > 1:
                keyname += str(k + 1)
            input.append(batch[keyname].to(self.device))
        label = batch["label"]
        gt_label = batch["gt_label"]
        index = batch["index"]
        impath = batch["impath"]
        label_onehot = create_onehot(label, self.num_classes).to(self.device)
        label = label.to(self.device)
        gt_label = gt_label.to(self.device)
        return input, label, index, impath, label_onehot, gt_label