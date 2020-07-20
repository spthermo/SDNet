# SDNet
Spatial Decomposition Network (SDNet) for content (anatomy) and style (modality) disentanglement.

This repo delivers the PyTorch implementation of the SDNet model presented in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841519300684). The original SDNet is implemented in Keras by the first author of the paper [Agis85](https://github.com/agis85/anatomy_modality_decomposition). This version of SDNet focuses on the comparison between spatial and vectorized latent space for the anatomy encoding (many variants are included). To actually compare the different variants, the segmentation task is adopted, using the ACDC cardiac imaging dataset (as in the original paper).
