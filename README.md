## Denoising Spatially Resolved Transcriptomics with Consistence of Heterogeneous Spatial Coordinates, Transcription and Morphology
Haiyue Wang, Shaoqing Feng, Peng Gao, and Xiaoke Ma
## Abstract
Spatially resolved transcriptomics (SRT) simultaneously captures the spatial coordinates, pathology, and transcription of cells within intact tissues, providing unprecedent opportunities for exploiting structure of tissues. However,  the extra-ordinary procedures in SRT bring in severe noise, hampering down-stream tasks, where algorithms for modeling and removing noise in SRT data is critical needed. Here, a deep autoencoder coupled with a self-supervised learning mechanism is developed to impute accurate transcription profiles of cells, jointly performing feature denoising and consistent learning to eliminate data heterogeneity and noise. As a result, MvDST reliably and precisely delineates tissue subgroups from simulated datasets under different perturbations. Moreover, it effectively distinguishes tumor-associated domains, identifies genetic markers within tumor regions, and uncovers intra-tumoral heterogeneity in cancer tissues. Furthermore, the robust performance of MvDST is verified across a variety of datasets generated from multiple platforms, such as Visium, STARmap, and osmFISH. Overall, de-noising via MvDST is anticipated to become a crucial initial step for analyzing spatially resovled transcriptomics data.
## Prerequisites
Machine with 16 GB of RAM. (All datasets tested required less than 16 GB). No non-standard hardware is required.
Python supprt packages (Python 3.9.0): For more details of the used package, please refer to 'requirements.txt' file
## File Descriptions
utils.py: Auxiliary functions for the MvDST model.

model.py: Base code for construct MvDST model.

train_nohistology.py: without histology information MvDST model.

image_feature.py: Extract morphological information from histology image.

## Tutorial
A jupyter Notebook of the tutorial for 10 $x$ Visium is accessible from : https://github.com/.
## Compared spatial domain identification algorithms
Algorithms that are compared include:

[SCANPY](https://github.com/scverse/scanpy-tutorials)

[SEDR](https://github.com/JinmiaoChenLab/SEDR/)

[SpaGCN](https://github.com/jianhuupenn/SpaGCN)

[DeepST](https://github.com/JiangBioLab/DeepST)

[STAGATE](https://github.com/zhanglabtools/STAGATE)

[stLearn](https://github.com/BiomedicalMachineLearning/stLearn)

[MIST](https://github.com/linhuawang/MIST)

[SPROD](https://github.com/yunguan-wang/SPROD)

[DIST](https://github.com/zhaoyp1997/DIST)

[Giotto](https://github.com/drieslab/Giotto)

[BayesSpace](https://github.com/edward130603/BayesSpace)

[GraphST](https://github.com/JinmiaoChenLab/GraphST)
## Contcat
Please send any questions or found bugs to Xiaoke Ma xkma@xidian.edu.cn.

