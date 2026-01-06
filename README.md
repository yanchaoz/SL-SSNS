# SL-SSNS

Official implementation of the *iScience* paper  
**A Distribution-aware Semi-Supervised Pipeline for Cost-effective Neuron Segmentation in Volume Electron Microscopy**  

📄 Paper: https://www.sciencedirect.com/science/article/pii/S2589004225027683

---

## Overview

Semi-supervised learning offers a cost-effective approach for neuron segmentation in electron microscopy (EM) volumes. This technique leverages unlabeled data to regularize supervised training for robust neuron boundary prediction. However, distribution mismatch between labeled and unlabeled data, caused by limited annotations and diverse neuronal structures, limits model generalization. In this study, we develop a distribution-aware pipeline to address the inherent mismatch issue and enhance semi-supervised neuron segmentation in EM volumes. At the data level, we select representative sub-volumes for annotation using an unsupervised measure of distributional similarity, ensuring broad coverage of neuronal structures. At the model level, we encourage consistent predictions across mixed views of labeled and unlabeled data. This design prompts the network to align feature distributions and learn shared semantics. Experiments on diverse EM datasets demonstrate the effectiveness of our method, which holds the potential to reduce proofreading demands and accelerate large-scale connectomic reconstruction efforts.

This repository contains the official implementation used in the *iScience* publication and can be readily adapted to other volumetric EM datasets.

---

## 🧪 Demo: Subvolume Selection

To help users better understand and apply our method, we provide an interactive demo by Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPYYeaycpdQjDiu_TQD4LqQbjezf40yc?usp=sharing), showcasing the **subvolume selection process** in both:

- The **spatial domain**
![image](https://github.com/user-attachments/assets/30a81673-03c6-4fd5-af86-c4d0ab60c2e5)
- The **embedding domain**
![image](https://github.com/user-attachments/assets/a9028640-b4e4-48cc-9bb9-cf3cbebbe9f0)

This demo can be readily adapted to your own EM datasets.

## 📦 Semi-supervised Pipeline
### Selective Labeling
##### Pretraining (If you want to retrain a model)
```
cd Pretraining
```
```
python pretraining.py
```
#### CGS Selection for EM Sub-volumes
```
cd CGS
```
```
python CGS.py
```
### Semi-supervised Training
```
cd IIC-Net
```
#### Supervised Warm-up
```
python warmup.py
```
#### Mixed-view Consistency Regularization
```
python semi_tuning.py
```
## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf) and [funkey/waterz](https://github.com/funkey/waterz). Should you have any further questions, please let us know. Thanks again for your interest.
