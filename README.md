![MIXXX_panel](./assets/mixxx.png)

## 0. Dataset Overview
The dataset is crawled from Musinsa, focusing on categories like hoodies, coats, and jeans. Using the crawled data, we generated descriptive captions with GPT-40 Mini.

The dataset is structured in the format (snapshot, product image, descriptive caption) to align with the FashionIQ dataset, which is commonly used for model training in fashion-related tasks.

Dataset has the following structure:

```
project_base_path
└─── musinsa
     ├─── men
     │    ├─── captions
     │    └─── image_splits
     ├─── women
     │    ├─── captions
     │    └─── image_splits
     ├─── goods
     │    ├─── men
     │    │    ├─── coat
     │    │    ├─── hoodie
     │    │    └─── jeans
     │    └─── women
     │         ├─── coat
     │         ├─── hoodie
     │         └─── jeans
     └─── snap
          ├─── men
          │    ├─── coat
          │    ├─── hoodie
          │    └─── jeans
          └─── women
               ├─── coat
               ├─── hoodie
               └─── jeans

```


## 1. Make Dataset
1. Crawling the dataset<br>

- Run the below file for snapshot
  - crawling/snapshot-crawling.ipynb
- Run the below file for goods
  - crawling/goodimage-crawling.ipynb<br>

2. After generating captions using GPT, structure the dataset.
- Run the file
  - crawling/make_data.ipynb




## 2. Environment Setup
For venv users
```
python3.10 -m venv .MIXXX
source .MIXXX/bin/activate
pip3 install fastapi uvicorn
```

For conda users
```
conda create -n MIXXX python==3.10
conda create MIXXX
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
pip install fastapi uvicorn 
```


## 3. Training
```
python src/blip_fine_tune_2.py
```




## 4. Inference 
```sh
python src/blip_validate.py
```



## 5. Web page 
```
#Terminal 1 for FE
npm start

  
#Terminal 2 for BE
uvicorn main:app --reload 
```

## 6. Checkpoints
Google drive:



## 7. Acknowledgement
Our implementation and development is based on [https://github.com/chunmeifeng/SPRC]


## 8. Members
**김준석** Data Preprocessing, Training & Inference Code Setting<br>
**문재원** Inference Code setting, Web Site Development(BE)<br>
**윤상민** Data crawling, Web Site Development(FE)<br>
**손채원** Web design<br>










