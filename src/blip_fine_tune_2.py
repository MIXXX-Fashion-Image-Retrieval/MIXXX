# from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torch.optim.lr_scheduler import OneCycleLR
import os

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset, MusinsaDataset, MusinsaRefDataset
from utils import collate_fn, update_train_running_results,update_train_running_results_dict, set_train_bar_description_dict,set_train_bar_description, extract_index_blip_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device
from validate_blip import compute_cirr_val_metrics, compute_fiq_val_metrics


def clip_finetune_musinsa(train_dress_types: List[str], val_dress_types: List[str],
                          num_epochs: int, blip_model_name: str, backbone: str, learning_rate: float, batch_size: int,
                          validation_frequency: int, transform: str, save_training: bool, save_best: bool, save_memory=True,
                          hyper_params=None, **kwargs):
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

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/clip_finetuned_on_fiq_{blip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"save-memory-in: {save_memory}")
    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(hyper_params, file, sort_keys=True, indent=4)
    blip_model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=device)
    update_method = getattr(blip_model, '_update_f_former', None)
    if callable(update_method):
        blip_model._update_f_former()

    input_dim = 224

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    classic_val_ref_datasets = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = MusinsaDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = MusinsaDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)
        classic_val_ref_dataset = MusinsaRefDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_ref_datasets.append(classic_val_ref_dataset)

    # Define the train datasets and the combining function
    relative_train_dataset = MusinsaDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=kwargs['num_workers'], pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, blip_model.parameters()), 'lr': learning_rate,
        #   'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay':0.05}])
        'betas': (0.9, 0.98), 'eps': 1e-7, 'weight_decay':0.05}])
    # scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1/50, steps_per_epoch=len(relative_train_loader), epochs=80)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, pct_start=1.5/num_epochs, div_factor=100., steps_per_epoch=len(relative_train_loader), epochs=num_epochs)

    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()


    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        train_running_results = {'images_in_epoch': 0}
        train_bar = tqdm(relative_train_loader, ncols=150)
        for idx, (reference_images, target_images, captions) in enumerate(train_bar):
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_images.to(device, non_blocking=True)
            print(reference_images.shape[0])
            target_images = target_images.to(device, non_blocking=True)

            # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            captions = generate_randomized_fiq_caption(flattened_captions)
            captions = [txt_processors["eval"](caption) for caption in captions]
            blip_model.train()
            # Extract the features, compute the logits and the loss
            with torch.cuda.amp.autocast():
                loss_dict = blip_model({"image":reference_images, "target":target_images, "text_input":captions})
                loss = 0.
                for key in loss_dict.keys():
                    loss += loss_dict[key]

            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_train_running_results_dict(train_running_results, loss_dict, images_in_batch)
            set_train_bar_description_dict(train_bar, epoch, num_epochs, train_running_results)

        loss_log_dict = {'epoch': epoch}
        for key in train_running_results.keys():
            if key != 'images_in_epoch':
                loss_log_dict[key] = float(
            train_running_results[key] / train_running_results['images_in_epoch'])
            # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
                pd.DataFrame(data=loss_log_dict, index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            blip_model.eval()
            recalls_at10 = []
            recalls_at50 = []

            # Compute and log validation metrics for each validation dataset (which corresponds to a different
            # FashionIQ category)
            for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                        idx_to_dress_mapping):
                index_features_ref, index_names_ref = extract_index_blip_features(classic_val_ref_dataset, blip_model, save_memory)
                index_features, index_names = extract_index_blip_features(classic_val_dataset, blip_model, save_memory)
                # 튜플을 리스트로 변환 후 수정
                index_features = list(index_features)
                index_features[-1] = index_features_ref[-1]
                # 필요시 다시 튜플로 변환
                index_features = tuple(index_features)

                recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                                    index_features, index_names, index_names_ref, txt_processors, save_memory)
                
                recalls_at10.append(recall_at10)
                recalls_at50.append(recall_at50)
                torch.cuda.empty_cache()

            results_dict = {}
            for i in range(len(recalls_at10)):
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
            results_dict.update({
                f'average_recall_at10': mean(recalls_at10),
                f'average_recall_at50': mean(recalls_at50),
                f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
            })

            print(json.dumps(results_dict, indent=4))
          
            # Validation CSV logging
            log_dict = {'epoch': epoch}
            log_dict.update(results_dict)
            validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
            validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_clip_best', epoch, blip_model, training_path)




def main():
    training_hyper_params = {
        "num_epochs": 20,                   # --num-epochs '30'
        "num_workers": 4,                   # --num-workers 4
        "blip_model_name": "blip2_cir_align_prompt",  # --blip-model-name 'blip2_cir_align_prompt'
        "backbone": "pretrain",             # 기본값
        "learning_rate": 2e-5,              # --learning-rate '2e-5'
        "batch_size": 4,                    # --batch-size 4
        "validation_frequency": 1,          # --validation-frequency 1
        "transform": "targetpad",           # --transform targetpad
        "target_ratio": 1.25,               # --target-ratio 1.25
        "save_training": True,              # --save-training
        "save_best": True,                  # --save-best
        "data_path": "./musinsa",           # 기본값 (필요시 변경)
        "loss_rtc": 0.4,                    # 기본값
        "loss_align": 0.4,                  # 기본값
        "loss_itm": 1,                      # 기본값
        "save_memory": True
    }

    
    training_hyper_params.update({
        'train_dress_types': ['jeans'],
        'val_dress_types': ['jeans']
    })
    
    # clip_finetune_musinsa 호출 시 하이퍼파라미터 딕셔너리를 추가 인자로 전달합니다.
    clip_finetune_musinsa(**training_hyper_params, hyper_params=training_hyper_params)
