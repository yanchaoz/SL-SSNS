# SL-SSNS
Official implementation of "Addressing Distribution Mismatch for Effective Semi-Supervised Neuron Segmentation" [under review]
## Selective Labeling
### Pretraining
```
cd Pretraining
```
```
python pretraining.py
```
### CGS Selection
```
cd CGS
```
```
python CGS.py
```
## Semi-supervised Training
```
cd IIC-Net
```
### Supervised Warm-up
```
python warmup.py
```
### Mixed-view Consistency Regularization
```
python semi_tuning.py
```
## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf) and [funkey/waterz](https://github.com/funkey/waterz). Should you have any further questions, please let us know. Thanks again for your interest.
