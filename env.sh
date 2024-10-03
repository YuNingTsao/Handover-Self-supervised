#!/bin/bash

source ~/.bashrc
#install anaconda_2019.03

conda init
source ~/.bashrc

conda install mamba
mamba init
source ~/.bashrc


#install ps-mt
mamba env create -f ps-mt.yaml

#install ssl
mamba env create -f ssl.yaml

#install transunet
mamba env create -f transunet.yaml

#install unet
mamba env create -f unet.yaml



#activate ps-mt
conda activate ps-mt

