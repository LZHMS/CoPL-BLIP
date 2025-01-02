from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.engine import TRAINER_REGISTRY, TrainerX


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from blip.blip_itm import blip_itm

from datasets.data_manager import DPLDataManager

_tokenizer = _Tokenizer()
from trainers.loss import GeneralizedCrossEntropy


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
    def __init__(self, cfg, classnames, blip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DPL.N_CTX
        ctx_init = "a photo of a"
        n_ctx = len(ctx_init.split(" "))
        tokenizer = blip_model.tokenizer
        embeddings = blip_model.text_encoder.embeddings
        
        classnames = [name.replace("_", " ") for name in classnames]
        ctx_prompts = [ctx_init + " " + name + '.' for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        ctx_prompt = tokenizer(ctx_prompts, padding='max_length', truncation=True, max_length=35, 
                            return_tensors="pt")
        self.attention_mask = ctx_prompt.attention_mask
        with torch.no_grad():
            ctx_embedding = embeddings(input_ids=ctx_prompt.input_ids)

        self.register_buffer("token_prefix", ctx_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", ctx_embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        print(f'Initial context: "{ctx_init}"')
        print(f"Number of context words (tokens): {n_ctx}")

        ctx_vectors = torch.empty(n_ctx, self.token_suffix.shape[2])
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx_prefix = nn.Parameter(ctx_vectors)  # to be optimized
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DPL.CLASS_TOKEN_POSITION
    
    def forward(self):
        ctx = self.ctx_prefix
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prompts = torch.cat(
            [
                self.token_prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                self.token_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, blip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, blip_model)
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
            return itm_output    # samples_num * 2
       
        elif match_head == 'itc':
            text_output = self.text_encoder(encoder_embeds = prompts,
                                attention_mask = self.attention_mask.to(prompts.device),                    
                                return_dict = True, 
                                mode = 'text')                     
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    
            sim = image_feat @ text_feat.t()        
            return sim

@TRAINER_REGISTRY.register()
class DPL(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.GCE = GeneralizedCrossEntropy(q=0.5)

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

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        blip_model = load_blip_to_cpu(cfg)

        if cfg.TRAINER.DPL.PREC == "fp32" or cfg.TRAINER.DPL.PREC == "amp":
            # CLIP's default precision is fp16
            blip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, blip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, _ = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.DPL.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, match_head='itc')
                if self.cfg.TRAINER.DPL.GCE:
                    loss = self.GCE(output, label)
                else:
                    loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, match_head='itc')
            if self.cfg.TRAINER.DPL.GCE:
                loss = self.GCE(output, label)
            else:
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.model(input, match_head='itc')

