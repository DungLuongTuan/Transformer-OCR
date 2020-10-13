# Transformer OCR
Optical to text model based on [transformer](https://arxiv.org/abs/1706.03762)

## Introduction
This repository contains Transformer-OCR model following architecture from paper [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/abs/1704.03549) but replace LSTM decoder layer by Transformer layer.

## Setup
### Prerequisites
- Python 3.6.9
### Installation
```
pip3 install -r requirements.txt
```

## Data preparation
Store data in fsns format. Generate TF-Record files follow repo [attention OCR](https://github.com/tensorflow/models/tree/master/research/attention_ocr)
After generate TF-record, you got `data.train`, `data.valid`, `data.test` corresponding to train, valid, test record
Images in record files need 
## Configuration file

Script for training and inference uses parameters from `hparams.py`. You need to specify some important parameters:
- *train_record_path:* path to `data.train`
- *valid_record_path:* path to `data.valid`
- *charset_path:* path to character list, default in `charsets/charset_size=11.txt` for recognizing number only
- *image_shape:* shape of input images
- *base_model_name:* name of Convolution model, one in `InceptionV3`, `InceptionResNetV2`, `EfficientNetB0 to B7`
- *end_point:* last convolution layer from base model which extract features from

## Training
To start training, run command:
```
python3 train.py
```
By default, output models and logs path is `training_checkpoints`
To watch training process:
```
tensorboard --logdir=training_checkpoints/logs
```
## inference
To inference single image:
```
python3 inference.py --inp_path=<path_to_image> --ckpt_path=<path_to_ckpt_model>
```
Check saliency maps in `inference_logs` for attention accuracy