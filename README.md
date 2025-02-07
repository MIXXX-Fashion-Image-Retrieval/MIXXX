
# Sentence-level Prompts Benefit Composed Image Retrieval 【ICLR 2024, Spotlight】

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sentence-level-prompts-benefit-composed-image/image-retrieval-on-fashion-iq)](https://paperswithcode.com/sota/image-retrieval-on-fashion-iq?p=sentence-level-prompts-benefit-composed-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sentence-level-prompts-benefit-composed-image/image-retrieval-on-cirr)](https://paperswithcode.com/sota/image-retrieval-on-cirr?p=sentence-level-prompts-benefit-composed-image)


### Prerequisites

	
The following commands will create a local Anaconda environment with the necessary packages installed.

```bash
conda create -n cir_sprc -y python=3.9
conda activate cir_sprc
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```


### Data Preparation

To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

### Training


```sh
python src/blip_fine_tune_2.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --blip-model-name 'blip2_cir_align_prompt' \
   --num-epochs {'50' for CIRR, '30' for fashionIQ} \
   --num-workers 4 \
   --learning-rate {'1e-5' for CIRR, '2e-5' for fashionIQ} \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 
```

### Evaluation


```sh
python src/blip_validate.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --blip-model-name {trained model name} \
   --model-path {for path} 
```

### CIRR Testing


```sh
python src/cirr_test_submission.py \
   --blip-model-name {trained model name} \
   --model-path {for path} \
```

### Checkpoints
Onedrive: [sprc_cirr.pt](https://1drv.ms/u/s!Aj0q22vyiZbabxR_hjpQ_DmYX3s?e=k9fmt1), [sprc_fiq.pt](https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj)

BaiduCloud: https://pan.baidu.com/s/18196NRV0Cdbn5uPc3LIgwg, password: t1at

models based on vit-l: [sprc_cirr_vitl.pt](https://drive.google.com/file/d/1217sOHWtvBG3Roq2AoNvVDH6eCfmYBRs/view?usp=sharing), [sprc_fiq_vitl.pt](https://drive.google.com/file/d/11p95msuuTAU6Pej7d_a-KjUHHpP3kH8Y/view?usp=sharing)

### Todo

code and pre-trained weights for rerank model

### Acknowledgement
Our implementation is based on [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir) and [LAVIS](https://github.com/salesforce/LAVIS).























(판넬 사진)


## 0. Dataset Overview
The dataset is crawled from Musinsa, focusing on categories like hoodies, coats, and jeans. Using the crawled data, we generated descriptive captions with GPT-40 Mini.

The dataset is structured in the format (snapshot, product image, descriptive caption) to align with the FashionIQ dataset, which is commonly used for model training in fashion-related tasks.

Dataset has the following structure:

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...
```


## 1. Make Dataset
1. Crawling the dataset



2. After generating captions using GPT, structure the dataset



--------
## 2. Environment Setup
For venv users
```
python3.10 -m venv .MIXXX
source .MIXXX/bin/activate
pip3 install fastapi uvicorn   <- 이렇게 하던지, 아니면 requirements.txt를 만들던지. 
```

For conda users
```
conda create -n MIXXX python==3.10
conda create MIXXX
pip3 install fastapi uvicorn 
```

## 3. Training





## 4. Inference 



## 5. Web page 
```
#Terminal 1
npm start

#Terminal 2
uvicorn main:app --reload 
```

## 6. Checkpoints
Google drive:


## 7. Acknowledgement
Our implementation and development is based on [https://github.com/chunmeifeng/SPRC]

## 8. Members
김준석 Data crawling, Data Preprocessing, Inference Code Setting
문재원 Inference Code setting, Web Site Development(BE)
윤상민 Data crawling, Web Site Development(FE)
손채원 Web design










