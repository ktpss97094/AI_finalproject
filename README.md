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

請先進入程式所在的資料夾

```
cd './CoachAI-Challenge-IJCAI2023/Track 2_ Stroke Forecasting/src/'
```

### Hyperparameters

我們設定要調的參數為: area_num、 batch_size、 dim (shot_dim、area_dim、player_dim、encode_dim 這4項會相等)、 epochs、 lr <br>
若要更改hyperparameters的搜索範圍，可以直接修改[]中的值 (format: [最低: 最高]) <br>
可使用BO.py產生超參數:

```
python BO.py niter
```

其中參數niter為要iterate的次數，但他最後會跑 niter + 5次。
ex.  python BO.py 5 -> 會跑 5+5 = 10 iter

之後再將BO.py輸出的optimize hyperparameters值寫入train.py即可 (note: 除了 lr 以外都要取整數)。

### Train

```
python train.py --model_type 'ShuttleNet' --output_folder_name './model' --area_num area_num  --batch_size batch_size --area_dim area_dim  --player_dim player_dim --encode_dim encode_dim --shot_dim shot_dim --epochs epochs  --lr lr
```

ex. python train.py --model_type 'ShuttleNet' --output_folder_name './model' --area_num 5  --batch_size 75 --area_dim 50  --player_dim 50 --encode_dim 50 --shot_dim 50 --epochs 139  --lr 0.00032233700746945427

### Generate Predictions

```
python generator.py ./model
```

### Compute Evaluation Metric

```
mv ./model/prediction.csv ../data/prediction.csv
python evaluation.py
```

## Experiment Results

score為	2.6639063255。
![image](https://github.com/ktpss97094/AI_finalproject/assets/122603032/63082eb6-5016-43ff-aa21-b50bb4f754c3)
