# About this repo
Spatial Decomposition Network (SDNet) for content (anatomy) and style (modality) disentanglement.

This repo delivers the **PyTorch implementation** of the SDNet model presented in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841519300684). The original SDNet is implemented in Keras by the first author of the paper [Agis85](https://github.com/agis85/anatomy_modality_decomposition). This version of SDNet focuses on the **comparison between spatial and vectorized latent space** for the anatomy encoding (many variants are included). To actually compare the different variants, the segmentation task is adopted, using the ACDC cardiac imaging dataset (as in the original paper).

## Prerequisites
All coding and experiments were using the following setup:
* PyTorch 1.5.1
* Cuda 10.1
* Python 3.7.5
* [Visdom](https://github.com/facebookresearch/visdom) - loss plots, images, etc.
* Packages: nibabel, opencv-python, skimage

## Training
To see all the available training (hyper)parameters use:
```
python main.py -h
```

Available SDNet variants:
1. Original architecture - UNet to encode anatomy in spatial latent variable (Variant A)
   * Gumbel Softmax is used instead of the binarization module for the UNet output --> smoother Dice loss convergence and a 3% increase in the validation accuracy
2. A VAE is used to encode the anatomy in a vector latent space (Variant B)
3. A VAE is used to re-encode the spatial output of the UNet - VAE output is used by the segmentor and the decoder (Variant C)

### SDNet architecture - Variant A
<img src="./misc/images/sdnet.png" width="750">
Train the original SDNet model for 60 epochs and batch size 10 using:
```
python main.py --model_name sdnet --epochs 60 --batch_size 10 --name visdom_experiment_name --visdom
```

### SDNet architecture - Variant B
<img src="./misc/images/sdnet2.png" width="650">
Train the 2-VAE SDNet model for 60 epochs and batch size 10 using:
```
python main.py --model_name sdnet2 --epochs 60 --batch_size 10 --name visdom_experiment_name --visdom
```

### SDNet architecture - Variant C
<img src="./misc/images/sdnet3.png" width="750">
Train the UNet+VAE SDNet model for 60 epochs and batch size 10 using:
```
python main.py --model_name sdnet3 --epochs 60 --batch_size 10 --name visdom_experiment_name --visdom
```


## To Do
Since this is an "in-progress" repository there are some more stuff to be added:
* A combined test script
* A script to visualize anatomy channels and modality (style) traversals of any model
* Add SPADE decoder implementation (now only AdaIN is available)

## Acknowledgements
Thank you [Agis85](https://github.com/agis85/anatomy_modality_decomposition) for the discussions and the original (Keras) implementation. Also thanks [Naoto Inoue](https://github.com/naoto0804) for the PyTorch implementation of the AdaIN module. 
