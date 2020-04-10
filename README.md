[![LinkedIn][linkedin-shield]][linkedin-url]
[![Stackoverflow][stackoverflow-shield]][stackoverflow-url]
[![Essayshark][essayshark-shield]][essayshark-url]

<!-- ABOUT THE PROJECT -->
## About The Project
              
Originally the MonuSeg dataset consists of 30 training images, each of size 1000 x 1000 in RGB format with its Ground Truth Masks as Single Channel Binary Images. In addition to these, there are 14 testing images of the same dimensions.
Moreover, I extracted patches of size 256 x 256 from the MonuSeg Dataset with 50% overlapping resulting in total 1080 patches.

### Network Diagrams

_**Network Summary of these diagrams can be seen in the ipython notebooks**_

### UNET
![UNET Architecture](https://www.researchgate.net/profile/Alan_Jackson9/publication/323597886/figure/fig2/AS:601386504957959@1520393124691/Convolutional-neural-network-CNN-architecture-based-on-UNET-Ronneberger-et-al.png)

### SEGNET
![SegNet Architecture](https://www.researchgate.net/profile/Vijay_Badrinarayanan/publication/283471087/figure/fig1/AS:391733042008065@1470407843299/An-illustration-of-the-SegNet-architecture-There-are-no-fully-connected-layers-and-hence.png)

### DEEPLABv3 +
![DeepLab v3+ Architecture](https://www.researchgate.net/profile/Manu_Goyal9/publication/330871054/figure/fig3/AS:722795042455552@1549339175407/Detailed-architecture-of-DeeplabV3-for-segmentation-on-skin-lesion-dataset-25.ppm)

<!-- GETTING STARTED -->

## Getting Started
Load and run the notebooks given in the repo in Google Collab. 

### Obtaining Data
Data publically available will be downloaded automatically from google drive using gdown. 

### Pre-Requisites
Libraries used in the project are:
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org)
* [Keras](https://keras.io)
* [Tensorflow](https://tensorflow.org)
* Google colab (Although not recommended)

<!-- USAGE EXAMPLES -->
## Usage
The LoadModel notebooks automatically load the model given you download the respective model given in the repository. The Deeplabv3 model was heavy so that is kept on the google drive, made public and LoadModel notebook deeplab will automatically download that in the current directory using gdown

### Config File
There are 3 config files for each model where you can set the hyperparameters of the respective model. I had to make 3 separate config files of each model because of the difference in the versions of the dependencies.

#### * (i)Qualitative Results (i)Quantitative Results (ii)Training and Validation graphs*


### Qualitative Results

#### SegNet
![SegNet Results][segnet-qual-res]

#### UNET
![UNET Results][unet-qual-res]

#### Deep Lab v3+
![DeepLab Results][deeplab-qual-res]

### Quantitative Results
| Model | Accuracy | Dice Coefficient | F1 Score | Binary Cross Entropy Loss|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SegNet | 89.89% | 0.6551 | 0.7301 | **0.2740** |
| UNET | 90.45% | 0.7218 | **0.7738** | 0.2474 |
| DeepLabV3+ | **90.79%** | **0.7707** | **0.7817** | 0.3670 |


### Training and Validation Graphs

#### UNET
![UNET Training Loss][unet-train-graph]
#### SegNet
![SegNet Training Loss][segnet-train-graph]
#### Deep Lab v3
![DeepLab Training Loss][deeplab-train-graph]


## Authors
* **Saqib Naseeb** - *Initial work* -

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/saqib-naseeb/
[stackoverflow-shield]:https://img.shields.io/badge/stackover-flow-orange
[stackoverflow-url]:https://stackoverflow.com/users/4938828/saqib-naseeb
[essayshark-shield]:https://img.shields.io/badge/Essay-Shark-blue
[essayshark-url]:https://essayshark.com/writers/amiablesyed.html

[unet-qual-res]: unet.png
[segnet-qual-res]: segNet_GT.png
[deeplabv3+-qual-res]: deeplabv3.png
