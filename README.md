# Adaptive Multi-Scale Transformer with Unified Attention for Enhanced Scene Segmentation in Autonomous Driving
Thank you for your interest in our research.This research is at the submission stage and the target journal is The Visual Computer.The pre-trained models will be made available after our paper is officially accepted.

## 📌 Introduction
Research in semantic segmentation for autonomous perception has advanced significantly with the emergence of deeplearning approaches. While Vision Transformers have revolutionized computer vision by introducing attention mechanisms,theircomputational overhead remains a challenge for real-world applications. We present AMST, a novel framework for efficient semantic segmentation incorporating two key innovations: 1) An adaptive attention mechanism that learns to focus on task relevant features dynamically, reducing computational complexity while maintaining high accuracy across diverse scenarios. 2) A multi-scale feature fusion module that effectively combines information from different resolution levels. For example, The proposed approach achieves superior performance on the Cityscapes benchmark, attaining 79.88% and 79.82% mIoU on the validation and test sets respectively. Meanwhile, Our model is trained on the ADE20K dataset and can maintain excellent segmentation performance when tested in actual campus scenes, proving its good generalization ability.
 
## 🚀 Installation
This repository is built upon [segmentation](https://github.com/open-mmlab/mmsegmentation) . Users can refer to the official link for installation. 

### 1️⃣ Environments:
- Python 3.8+
- PyTorch 2.1.2
- CUDA 11.8
- mmcv 2.1.0

### 2️⃣ Installation
git clone 

cd amst

pip install -r requirements.txt

##  Data Preparation
1.Download the offical dataset data

2.Process data according to the official [MMSEG documentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/2_dataset_prepare.html)

## 🚀 Train

🎯 Navigate to the project directory

cd AMST

🎯 Train with a single GPU

 python tools/train.py configs/amst/amst_cityscapes-1024x1024.py

## 🚀 Test

🎯 Test with a single GPU

 python tools/test.py configs/amst/amst_cityscapes-1024x1024.py

