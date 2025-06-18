## Denoising Spatially Resolved Transcriptomics with Consistence of Heterogeneous Spatial Coordinates, Transcription and Morphology
Haiyue Wang, Shaoqing Feng, Peng Gao, and Xiaoke Ma
## Abstract
Spatially resolved transcriptomics (SRT) simultaneously captures spatial coordinates, pathological features, and transcriptional profiles of cells within intact tissues, offering unprecedented opportunities to explore tissue architecture. However, SRT data often suffer from substantial technical noise introduced by experimental procedures, posing challenges for downstream analyses. To overcome these challenges, we introduce a Multi-view Denoising framework for Spatial Transcriptomics (MvDST), which integrates a deep autoencoder and self-supervised learning to jointly reconstruct expression profiles, denoise features, and ensure cross-view consistency, effectively reducing technical noise and heterogeneity.  As a result, MvDST reliably and accurately delineates tissue subgroups across simulated datasets under various perturbations. In real cancer datasets, it distinguishes tumor-associated domains, identifies region-specific marker genes, and reveals intra-tumoral heterogeneity. Furthermore, we validate the robustness of MvDST across multiple spatial transcriptomics platforms, including 10 $\times$ Visium, STARmap, and osmFISH. Overall, these results demonstrate that MvDST can serve as a crucial initial step for analyzing spatially resolved transcriptomics data.
## Prerequisites
Machine with 16 GB of RAM. (All datasets tested required less than 16 GB). No non-standard hardware is required.
Python supprt packages (Python 3.9.0): For more details of the used package, please refer to 'requirements.txt' file
## File Descriptions
utils.py: Auxiliary functions for the MvDST model.

model.py: Base code for construct MvDST model.

train_nohistology.py: without histology information MvDST model.

image_feature.py: Extract morphological information from histology image.

## Tutorial
A jupyter Notebook of the tutorial for 10 $x$ Visium is accessible from : https://github.com/zilanjiuwan/MvDST/blob/main/Tutorial/DLPFC.ipynb.

Tutorial notebook for using MvDST integrate morphological features extracted by different models is available at: https://github.com/zilanjiuwan/MvDST/blob/main/Tutorial/Breast_Cancer.ipynb.

MvDST is applicable to imaging-based ST Platform:https://github.com/zilanjiuwan/MvDST/blob/main/Tutorial/STARmap.ipynb.
## System Requirements
Python support packages (Python 3.9.18):
scanpy, igraph, pandas, numpy, scipy, scanpy, anndata, sklearn, seaborn, torch, tqdm.
For more details of the used package., please refer to 'requirements.txt' file.
The coding here is a generalization of the algorithm given in the paper. MvDST is written in Python programming language. 
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
Please send any questions or found bugs to Haiyue Wang haiyue_wang1223@163.com.


