# NeuroBench_LS: Benchmarking Foundation Models on a Lifespan-Scale Brain MRI Dataset

 

<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

**💡 NeuroBench-LS **, a unified benchmark framework designed to evaluate and enhance the generalizability of foundation models across the human lifespan using brain MRI. *** 

## 📋 Our Dataset

We provide a comprehensive MRI dataset on [Harvard Dataverse](https://dataverse.harvard.edu/) spanning the full human lifespan—from fetus and infant to elderly adults. The collection includes over **22,000** high-quality **T1-weighted** and **T2-weighted** structural brain MRI scans.

- **Publicly released subcortical segmentation** are available via the official [Dataverse link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B5OU7H).
- **Private or restricted datasets** can be accessed upon request. Please contact the corresponding principal investigator (PI) for access approval.

For more details on data usage, structure, and benchmarks, please refer to the full documentation in this repository.   

## ✓ Get Dataset

We provide links and instructions for all dataset

1. **ABCD** : [Adolescent Brain Cognitive Development Study](https://nda.nih.gov/general-query.html?q=query=featured-datasets:Adolescent%20Brain%20Cognitive%20Development%20Study%20(ABCD))

   **License**: NDA (Must apply through the NIH's NIMH Data Archive)

2. **ABIDE** : [Autism Brain Imaging Data Exchange](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)

   **License**: Freely available to the research community. No DUA Required.

3. **ADHD 200** : [ADHD-200 Global Competition Dataset](https://fcon_1000.projects.nitrc.org/indi/adhd200/)

   **License**: Freely available to the research community. No DUA Required.

4. **ADNI** : [Alzheimer's Disease Neuroimaging Initiative](https://ida.loni.usc.edu/login.jsp?project=ADNI&page=HOME)

   **License**: Application Required via LONI IDA (Create an account and  request access at https://ida.loni.usc.edu)

5. **BCP** : [Baby Connectome Project](https://nda.nih.gov/edit_collection.html?id=2848)

   **License**: NDA (Must apply through the NIH's NIMH Data Archive)

6. **dHCP** : [Developing Human Connectome Project](https://www.developingconnectome.org/data-release/second-data-release/)

   **License**: Distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

7. **HBCD** : [Healthy Brain and Child Development Study](https://hbcd-docs.readthedocs.io/data_access/)

   **License**: NDA (Must apply through the NIH's NIMH Data Archive)

8. **HBN** : [Healthy Brain Network](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_EEG.html#Direct%20Down)

   **License**: NDA (Must apply through the NIH's NIMH Data Archive)

9. **HCP-A** : [Human Connectome Project – Aging](https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases)

   **License**: NDA (Must apply through the NIH's NIMH Data Archive)

10. **HCP-D** : [Developing Human Connectome Project-Devolpment](https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases)

    **License**: NDA (Must apply through the NIH's NIMH Data Archive)

11. **PING** : [Pediatric Imaging, Neurocognition, and Genetics Study](https://nda.nih.gov/edit_collection.html?id=2607)

    **License**: NDA (Must apply through the NIH's NIMH Data Archive)

12. **FeTA**:  [Fetal Tissue Annotation Dataset](https://www.synapse.org/Synapse:syn23747212/wiki/608434)

    **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

13. **MOMMA**: Not publish

14. **FIND**: Not publish

15. **Subcortical Segmentagtion** : [Brain Image](https://nda.nih.gov/study.html?id=1745)

    

    If you are using Python, this means providing a `requirements.txt` file (if using `pip` and `virtualenv`), providing `environment.yml` file (if using anaconda), or a `setup.py` if your code is a library. 

It is good practice to provide a section in your README.md that explains how to install these dependencies. Assume minimal background knowledge and be clear and comprehensive - if users cannot set up your dependencies they are likely to give up on the rest of your code as well. 

If you wish to provide whole reproducible environments, you might want to consider using Docker and upload a Docker image of your environment into Dockerhub. 

#### 2. Installation

We procide`requirements.txt`in each bench. All model under PyTorch-GPU.

#### 3. Evaluation code

We provide eval.py for each bench.



## 🎉 NeuroBench_LS

### Mask Autoencoder for Lifespan

We provide 2D and 3D MAE

1. [2D MAE](https://github.com/rlu25/NeuroBrench_LS/tree/main/MAE_Model/MAE_2D) - 2D MAE pretrained on slice-based MRI acquisitions.
2. [3D MAE](https://github.com/rlu25/NeuroBrench_LS/tree/main/MAE_Model/MAE_3D) - 3D MAE pretrained on volumetric MRI scans
3. [Google Drive](https://drive.google.com/drive/folders/1JwdIB00tUwrLIUAX7HIbTgUj6c97j6Aa?usp=sharing) - Weights can be downloaded here

### Baseline

We provide seven SOTA for different tasks

1. [MAPSeg](https://github.com/XuzheZ/MAPSeg) - Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling
1. [QuickNAT](https://github.com/ai-med/QuickNATv2) - QuickNAT: Segmenting MRI Neuroanatomy
1. [FastSurfer](https://github.com/Deep-MI/FastSurfer) - A fast and accurate deep-learning based neuroimaging pipeline
1. [cGAN](https://github.com/icon-lab/pGAN-cGAN) - pGAN and cGAN models for T1 to T2 synthesis
1. [PTNet](https://github.com/XuzheZ/PTNet) - Pytorch implementation of PTNet for high-resolution and longitudinal infant MRI synthesis.
1. [ORDER](https://github.com/jaygshah/Robust-Brain-Age-Prediction) - obust Brain Age Prediction
1. [TSAN](https://github.com/Milan-BUAA/TSAN-brain-age-estimation) - Brain Age Estimation From MRI Using Cascade Networks with Ranking Loss

### MAE_finetune

1. [Subcortical Segmentation](https://github.com/rlu25/NeuroBrench_LS/tree/main/Benchmark_Finetune/Subcortical%20Segmentation)

2. [MRI Synthesis](https://github.com/rlu25/NeuroBrench_LS/tree/main/Benchmark_Finetune/MRI%20Synthesis)

3. [Age Prediction](https://github.com/rlu25/NeuroBrench_LS/tree/main/Benchmark_Finetune/Age%20Prediction)

4. [Google Drive](https://drive.google.com/drive/folders/1JwdIB00tUwrLIUAX7HIbTgUj6c97j6Aa?usp=sharing) - Weights can be downloaded here

   

## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at rliu60@emory.edu or open an issue on this GitHub repository. 

All contributions welcome! 
