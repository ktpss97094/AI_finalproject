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

我們設定的hyperparameters為....**(TODO)**
若要產生其他的hyperparameters值，可使用BO.py產生超參數:

```
python BO.py niter
```

其中參數niter為要iterate的次數，通常設為5。

之後再將BO.py輸出的optimize hyperparameters值寫入train.py即可。

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

score為2.6901113591。