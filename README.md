# SL-SSNS
Official implementation of ["A Distribution-aware Semi-Supervised Pipeline for Cost-effective Neuron Segmentation in Volume Electron Microscopy"](https://www.biorxiv.org/content/10.1101/2024.05.26.595303v3).

## 🧪 Demo: Subvolume Selection

To help users better understand and apply our method, we provide an interactive demo by Colab, showcasing the **subvolume selection process** in both: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPYYeaycpdQjDiu_TQD4LqQbjezf40yc?usp=sharing)

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
