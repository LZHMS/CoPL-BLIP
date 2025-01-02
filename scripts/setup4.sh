#!/bin/bash

CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 0 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 2 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 4 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 6 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 8 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 10 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 12 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=2 bash scripts/parse.sh oxford_flowers ablation MixDivideWarmupAugmentationBLIP