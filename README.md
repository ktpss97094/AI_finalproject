# AI_finalproject

## Overview

這是國立陽明交通大學111學年度第二學期人工智慧概論課程的期末專題實作。內容為2023 IJCAI CoachAI Badminton Challenge的Track 2: Forecasting Future Turn-Based Strokes in Badminton Rallies。

## Prerequisite

### Coding Environment

* Python 3.9
* Nvidia GPU

### Packages Version

使用requirements.txt安裝我們使用的package版本:

```
pip install -r requirements.txt
```

## Usage

### Initialize Hyperparameters

我們設定要調的參數為: area_num、 batch_size、 dim (shot_dim、area_dim、player_dim、encode_dim)、 epochs、 lr 
若要產生更改hyperparameters的搜索範圍，可以直接修改[]中的值 (format: [最低: 最高])，
可使用BO.py產生超參數:

```
python BO.py niter
```

其中參數niter為要iterate的次數，但他最後會跑 niter + 5次。
ex.  python BO.py 5 -> 會跑 5+5 = 10 iter

之後再將BO.py輸出的optimize hyperparameters值寫入train.py即可(但除了 lr 以外都要取整數)。

### Train

```
python train.py --output_folder_name ".\model\" --model_type ShuttleNet --encode_length encode_length --seed_value seed_value
```

其中參數encode_length、seed_value填入hyperparameter值。

### Generate Predictions

```
python generator.py .\model
```

### Compute Evaluation Metric

```
mv .\model\prediction.csv ..\data\prediction.csv
python evaluation.py
```

## Experiment Results

score為	2.6639063255。
![image](https://github.com/ktpss97094/AI_finalproject/assets/122603032/63082eb6-5016-43ff-aa21-b50bb4f754c3)
