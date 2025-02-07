# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR


from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, MusinsaDataset, MusinsaRefDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics, compute_fiq_val_ranking


def clip_finetune_fiq(val_dress_types: List[str], blip_model_name, backbone, model_path):
    """
    Fine-tune CLIP on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning leanring rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned CLIP model
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best CLIP model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    blip_model, _, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    checkpoint_path = model_path

    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    print("Missing keys {}".format(msg.missing_keys))

    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    classic_val_ref_datasets = []


    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = MusinsaDataset('inference', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = MusinsaDataset('inference', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)
        classic_val_ref_dataset = MusinsaRefDataset('inference', [dress_type], 'classic', preprocess, )
        classic_val_ref_datasets.append(classic_val_ref_dataset)



    blip_model.eval()

    # Compute and log validation metrics for each validation dataset (which corresponds to a different
    # FashionIQ category)
    for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                idx_to_dress_mapping):
        index_features_ref, index_names_ref = extract_index_blip_features(classic_val_ref_dataset, blip_model)
        index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model)

        index_features = list(index_features)
        index_features[-1] = index_features_ref[-1]
        index_features = tuple(index_features)

        sorted_index_names = compute_fiq_val_ranking(relative_val_dataset, blip_model,
                                                            index_features, index_names, index_names_ref, txt_processors)
        
        
        torch.cuda.empty_cache()



        return sorted_index_names



# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
#     parser.add_argument("--blip-model-name", default="blip2_cir_rerank_learn", type=str)
#     parser.add_argument("--backbone", type=str, default="pretrain", help="pretrain for vit-g, pretrain_vitL for vit-l")
#     parser.add_argument("--model-path", type=str)

#     args = parser.parse_args()
#     if args.dataset.lower() not in ['fashioniq', 'cirr']:
#         raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

#     if args.dataset.lower() == 'cirr':
#         blip_validate_cirr(args.blip_model_name, args.backbone, args.model_path)
#     elif args.dataset.lower() == 'fashioniq':
#         clip_finetune_fiq(['dress', 'toptee', 'shirt'], args.blip_model_name, args.backbone, args.model_path)
def main():
    dataset = 'fashionIQ'
    blip_model_name = "blip2_cir_align_prompt"
    backbone = "pretrain"
    model_path = "models/men_hoodie/saved_models/tuned_clip_best.pt"

    sorted_index_names = clip_finetune_fiq(['hoodie'], blip_model_name, backbone, model_path)
    print(sorted_index_names)

if __name__ == '__main__':
    main()
