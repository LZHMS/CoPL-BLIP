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

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.utils import ( MetricMeter, AverageMeter, mkdir_if_missing, load_pretrained_weights )
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling.ops.utils import sharpen_prob, create_onehot

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from blip.blip_itm import blip_itm

#from datasets.data_manager import DPLDataManager
from datasets.data_manager import Dataloader_XU, Dataloader_eval

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


class FeaturedPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, features):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPL.N_CTX
        ctx_init = "a photo of a"
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
        prompts = [prompt_prefix + " " + name + ", " + features[name] + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        #self.suffix_ctx = nn.Parameter(torch.cat([embedding[:, 1 + n_ctx + nl + 1 : 1 + n_ctx + nl + 1 + 13, :] for nl in name_lens]))  # to be optimized
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.DPL.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        #suffix_ctx = self.suffix_ctx

        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            ctx_i = ctx[i : i + 1, :, :]
            class_i = suffix[i : i + 1, : name_len + 1, :]
            #suffix_ctx_i = suffix_ctx[i : i + 1, :, :]
            suffix_i = suffix[i : i + 1, name_len + 1 :, :]
            prompt = torch.cat(
                [
                    prefix_i,     # (1, 1, dim)
                    ctx_i,  # (1, n_ctx//2, dim)
                    class_i,      # (1, name_len, dim)
                    #suffix_ctx_i,  # (1, n_ctx//2, dim)
                    suffix_i,     # (1, *, dim)
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts


class FeaturedCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, features):
        super().__init__()
        self.prompt_learner = FeaturedPromptLearner(cfg, classnames, clip_model, features)
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
    

@TRAINER_REGISTRY.register()
class DPL(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE = GeneralizedCrossEntropy(q=0.5)
        self.temp = cfg.TRAINER.DPL.TEMP
        self.beta = cfg.TRAINER.DPL.BETA
        self.theta = 0.01
        self.co_lambda = cfg.TRAINER.DPL.CO_LAMBDA
        self.loss = deque(maxlen=5)
        self.match_probs = deque(maxlen=5)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPL.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader_XU(self, pred=[], prob=[]):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = Dataloader_XU(self.cfg, pred=pred, prob=prob)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains

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
        self.fmodel = FeaturedCLIP(cfg, classnames, clip_model, features)
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
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.foptim = build_optimizer(self.fmodel.prompt_learner, cfg.OPTIM)
        self.fsched = build_lr_scheduler(self.foptim, cfg.OPTIM)
        self.register_model("featured_prompt_learner", self.fmodel.prompt_learner, self.foptim, self.fsched)

        self.scaler = GradScaler() if cfg.TRAINER.DPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.fmodel = nn.DataParallel(self.fmodel)
            self.blip = nn.DataParallel(self.blip)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()
        self.build_data_loader_XU()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # co-divide
        if self.epoch % 5 == 0:
            self.match_ID, self.refined_labels, self.refined_labels_expand = self.eval_train()

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

    def eval_train(self):
        self.set_model_mode("eval")
        
        data_len = len(self.train_loader_x.dataset)
        #--- Step 1: do eval for splitting the dataset
        losses = torch.zeros(data_len)     # for GMM modeling
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, _ = self.parse_batch(batch)
                output_simple, output_featured = 0, 0
                for input_i in input:
                    output_simple_i = self.model(input_i)
                    output_featured_i = self.fmodel(input_i)
                    output_simple += output_simple_i
                    output_featured += output_featured_i       
                output_simple /= len(input)
                output_featured /= len(input)

                co_reg = kl_loss_compute(output_simple, output_featured, reduce=False) + kl_loss_compute(output_featured, output_simple, reduce=False)
                loss_simple = F.cross_entropy(output_simple, label, reduction='none')
                loss_featured = F.cross_entropy(output_featured, label, reduction='none')
                loss = loss_simple + loss_featured + self.co_lambda * co_reg
                for b in range(label.size(0)):
                    losses[index[b]] = loss[b]

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        self.loss.append(losses)

        if self.cfg.TRAINER.DPL.AVERAGE_LOSS:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(list(self.loss), dim=0)
            input_loss = history.mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean = gmm.means_.reshape(-1)
        std = gmm.covariances_.reshape(-1)
        idx_clean = mean.argmin()
        idx_noise = mean.argmax()

        mean_clean = torch.tensor(mean[idx_clean]).cuda()
        mean_noise = torch.tensor(mean[idx_noise]).cuda()
        std_clean = torch.tensor(std[idx_clean]).cuda()
        std_noise = torch.tensor(std[idx_noise]).cuda()

        # calculate the thredhold
        alpha_1 = mean_clean + torch.sqrt(-2 * (std_clean ** 2) * torch.log(self.theta * 
            std_clean * torch.sqrt(torch.tensor(2 * torch.pi)) + 1e-8))
        alpha_2 = mean_noise - torch.sqrt(-2 * (std_noise ** 2) * torch.log(self.theta *
            std_noise * torch.sqrt(torch.tensor(2 * torch.pi)) + 1e-8))
	
        print(f"alpha_1: {alpha_1} alpha_2: {alpha_2}")
        if alpha_1 > alpha_2:
            clean_ID = (input_loss < alpha_2.item())
            noisy_ID = (input_loss > alpha_1.item())
        else:
            clean_ID = (input_loss < alpha_1.item())
            noisy_ID = (input_loss > alpha_2.item())
        confused_ID = ~(clean_ID | noisy_ID)     # confusing samples

        # clean probalities for the label
        clean_prob = torch.tensor(prob[:, idx_clean], device=self.device).reshape(-1, 1)

        clean_ID = torch.nonzero(clean_ID, as_tuple=True)[0]
        noisy_ID = torch.nonzero(noisy_ID, as_tuple=True)[0]
        confused_ID = torch.nonzero(confused_ID, as_tuple=True)[0]
        
        #--- Step 2: do label refinement for the three subsets
        refined_labels = torch.zeros(data_len, dtype=torch.long)
        refined_labels_expand = torch.zeros((data_len, self.num_classes))
        itm_scores = torch.zeros(data_len)
        with torch.no_grad():
            for self.batch_id, batch in enumerate(self.train_loader_x):
                input, label, index, _, label_onehot = self.parse_batch(batch)

                output_simple, output_featured = 0, 0
                for input_i in input:
                    # simple prompt learning
                    output_simple_i = F.softmax(self.model(input_i), dim=1)
                    # featured prompt learning
                    output_featured_i = F.softmax(self.fmodel(input_i), dim=1)
                    output_simple += output_simple_i
                    output_featured += output_featured_i       
                output_simple /= len(input)
                output_featured /= len(input)

                probs = clean_prob[index]

                # label refinement
                refined_predict = (output_simple + output_featured) / 2
                refined_label = sharpen_prob(refined_predict, self.temp)

                # label mixrefinement
                mixrefined_predict = probs * label_onehot + (1 - probs) * (output_simple + output_featured) / 2 
                mixrefined_label = sharpen_prob(mixrefined_predict, self.temp)

                for i, id in enumerate(index):
                    if id in clean_ID:
                        # Label absorb of labeled samples
                        refined_labels[id] = label[i]
                        refined_labels_expand[id] = label_onehot[i]
                    elif id in noisy_ID:
                        # Label refinement for unlabeled data
                        refined_labels[id] = refined_label[i].argmax()
                        refined_labels_expand[id] = refined_label[i]
                    else:
                        # mixrefine confused samples
                        refined_labels[id] = mixrefined_label[i].argmax()
                        refined_labels_expand[id] = mixrefined_label[i]

                    #--- Step 3: do pesudo label evaluation
                    # discriminator
                    with torch.no_grad():
                        itm_score = 0
                        for input_i in input:
                            itm_score += self.blip(input_i[i].unsqueeze(dim=0), refined_labels[id].unsqueeze(dim=0))
                        itm_scores[id] = itm_score / len(input)
        
        itm_scores = (itm_scores - itm_scores.min()) / (itm_scores.max() - itm_scores.min())
        self.match_probs.append(itm_scores)

        if self.cfg.TRAINER.DPL.AVERAGE_MATCH:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(list(self.match_probs), dim=0)
            input_match_prob = history.mean(0)
            input_match_prob = input_match_prob.reshape(-1, 1)
        else:
            input_match_prob = itm_scores.reshape(-1, 1)

        # fit a two-component GMM to the match probality
        input_match_prob = input_match_prob.cpu()
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=5e-4)
        gmm.fit(input_match_prob)
        probs = gmm.predict_proba(input_match_prob)

        # divide the pseudo labels into match and unmatch set
        w = probs[:, gmm.means_.argmax()]
        match_ID = torch.tensor(w >= 0.5)

        match_ID = torch.nonzero(match_ID, as_tuple=True)[0]
        return match_ID, refined_labels, refined_labels_expand
    

    def forward_backward(self, batch):
        input, label, index, _, _ = self.parse_batch(batch)

        index = [index] * len(input)
        index = torch.cat(index, 0)
        input = torch.cat(input, 0)

        input_x, label_x = [], []
        input_u, label_u = [], []
        for i, id in enumerate(index):
            if id.item() in self.match_ID:
                input_x.append(input[i])
                label_x.append(self.refined_labels_expand[id.item()])
            else:
                input_u.append(input[i])
                label_u.append(self.refined_labels_expand[id.item()])

        match_empty = len(input_x) == 0
        unmatch_empty = len(input_u) == 0
        if not match_empty:
            input_x = torch.stack(input_x, dim=0)
            label_x = torch.stack(label_x, dim=0)
        if not unmatch_empty:
            input_u = torch.stack(input_u, dim=0)
            label_u = torch.stack(label_u, dim=0)

        if match_empty:
            all_inputs = input_u
            all_labels = label_u
        elif unmatch_empty:
            all_inputs = input_x
            all_labels = label_x
        else:
            all_inputs = torch.cat([input_x, input_u], dim=0)
            all_labels = torch.cat([label_x, label_u], dim=0)

        # mixmatch for the unmatch label set
        l = np.random.beta(self.beta, self.beta)
        l = max(l, 1 - l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        label_a, label_b = all_labels, all_labels[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_label = l * label_a + (1 - l) * label_b

        # get all samples
        # if not match_empty:
        #     all_inputs = torch.cat([input_x, mixed_input], dim=0).to(self.device)
        #     all_labels = torch.cat([label_x, mixed_label], dim=0).to(self.device)
        # else:
        all_inputs = mixed_input.to(self.device)
        all_labels = mixed_label.to(self.device)

        prec = self.cfg.TRAINER.DPL.PREC
        if prec == "amp":
            with autocast():
                output_simple = self.model(input)
                output_featured = self.fmodel(input)

                predict_simple = F.softmax(output_simple, dim=1)
                predict_featured = F.softmax(output_featured, dim=1)

                co_reg = kl_loss_compute(predict_simple, predict_featured) + kl_loss_compute(predict_featured, predict_simple)
                if self.cfg.TRAINER.DPL.GCE:
                    loss_simple = self.GCE(output_simple, label)
                    loss_featured = self.GCE(output_featured, label)
                else:
                    loss_simple = F.cross_entropy(output_simple, label)
                    loss_featured = F.cross_entropy(output_featured, label)
                loss = loss_simple + loss_featured + self.co_lambda * co_reg
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_simple = self.model(all_inputs)
            output_featured = self.fmodel(all_inputs)

            predict_simple = F.softmax(output_simple, dim=1)
            predict_featured = F.softmax(output_featured, dim=1)

            co_reg = kl_loss_compute(predict_simple, predict_featured) + kl_loss_compute(predict_featured, predict_simple)
                
            if self.cfg.TRAINER.DPL.GCE:
                loss_simple = self.GCE(output_simple, all_labels.argmax(dim=1))
                loss_featured = self.GCE(output_featured, all_labels.argmax(dim=1))
            else:
                loss_simple = F.cross_entropy(output_simple, all_labels)
                loss_featured = F.cross_entropy(output_featured, all_labels)
            loss = loss_simple + loss_featured + self.co_lambda * co_reg
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": (compute_accuracy(output_simple, all_labels.argmax(dim=1))[0].item() + compute_accuracy(output_featured, all_labels.argmax(dim=1))[0].item()) / 2,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def model_inference(self, input):
        return (self.model(input) + self.fmodel(input)) / 2
    
    def parse_batch(self, batch):
        input = []
        for k in range(self.cfg.DATALOADER.K):
            keyname = "img"
            if (k + 1) > 1:
                keyname += str(k + 1)
            input.append(batch[keyname].to(self.device))
        label = batch["label"]
        index = batch["index"]
        impath = batch["impath"]
        label_onehot = create_onehot(label, self.num_classes).to(self.device)
        label = label.to(self.device)
        return input, label, index, impath, label_onehot