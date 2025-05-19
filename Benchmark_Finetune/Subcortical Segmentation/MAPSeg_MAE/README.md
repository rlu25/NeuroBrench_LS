# MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D <ins>M</ins>asked <ins>A</ins>utoencoding and <ins>P</ins>seudo-Labeling

[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MAPSeg_Unified_Unsupervised_Domain_Adaptation_for_Heterogeneous_Medical_Image_Segmentation_CVPR_2024_paper.html) / [arXiv](https://arxiv.org/abs/2303.09373)

## Usage: 

    conda create --name mapseg --file requirements.txt
    conda activate mapseg

For training: 
    
    python train.py --config=YOUR_PATH_TO_YAML
Training procedure:
    
MAE pretraining: 
    I you need MAE model detail, please refer to [here](https://github.com/rlu25/NeuroBrench_LS/tree/main/MAE_Model/MAE_3D). 

For inference: 

    python test.py # be sure to edit the test.py 

# Cite:
If you found our work helpful, please cite our work:

    @InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Xuzhe and Wu, Yuhao and Angelini, Elsa and Li, Ang and Guo, Jia and Rasmussen, Jerod M. and O'Connor, Thomas G. and Wadhwa, Pathik D. and Jackowski, Andrea Parolin and Li, Hai and Posner, Jonathan and Laine, Andrew F. and Wang, Yun},
    title     = {MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {5851-5862}}
