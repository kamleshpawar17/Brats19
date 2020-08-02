# Brats19

This repo contains code for the Brain Tumor Segmentation Challenge 2020, published in Brainlesion 2019:

[Pawar K, Chen Z, Shah NJ, Egan GF. An Ensemble of 2D Convolutional Neural Network for 3D Brain Tumor Segmentation. *In International MICCAI Brainlesion Workshop 2019 Oct 17 (pp. 359-367). Springer, Cham.*](https://doi.org/10.1007/978-3-030-46640-4_34)

## Installation
Clone the repository and run setup.sh as
```
sh setup.sh
```

## Run
To run prediction, copy all four MRI contrast imges ending with names (*.t1.nii.gz, *.t2.nii.gz, *,t1ce.nii.gz, *.flair.nii.gz) into the directory **./data** then run the following
```
cd ./src
python predict_all_no_gt.py
```
The ouput will be in **./data/results/**

