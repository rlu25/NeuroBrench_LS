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

15. **Subcortical Segmentagtion** : [Brain subcortical segmentation](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B5OU7H)[Brain Image](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B5OU7H)https://nda.nih.gov/study.html?id=1745

#### 1. Specification of dependencies

If you are using Python, this means providing a `requirements.txt` file (if using `pip` and `virtualenv`), providing `environment.yml` file (if using anaconda), or a `setup.py` if your code is a library. 

It is good practice to provide a section in your README.md that explains how to install these dependencies. Assume minimal background knowledge and be clear and comprehensive - if users cannot set up your dependencies they are likely to give up on the rest of your code as well. 

If you wish to provide whole reproducible environments, you might want to consider using Docker and upload a Docker image of your environment into Dockerhub. 

#### 2. Training code

Your code should have a training script that can be used to obtain the principal results stated in the paper. This means you should include hyperparameters and any tricks that were used in the process of getting your results. To maximize usefulness, ideally this code should be written with extensibility in mind: what if your user wants to use the same training script on their own dataset?

You can provide a documented command line wrapper such as `train.py` to serve as a useful entry point for your users. 

#### 3. Evaluation code

Model evaluation and experiments often depend on subtle details that are not always possible to explain in the paper. This is why including the exact code you used to evaluate or run experiments is helpful to give a complete description of the procedure. In turn, this helps the user to trust, understand and build on your research.

You can provide a documented command line wrapper such as `eval.py` to serve as a useful entry point for your users.

#### 4. Pre-trained models

Training a model from scratch can be time-consuming and expensive. One way to increase trust in your results is to provide a pre-trained model that the community can evaluate to obtain the end results. This means users can see the results are credible without having to train afresh.

Another common use case is fine-tuning for downstream task, where it's useful to release a pretrained model so others can build on it for application to their own datasets.

Lastly, some users might want to try out your model to see if it works on some example data. Providing pre-trained models allows your users to play around with your work and aids understanding of the paper's achievements.

#### 5. README file includes table of results accompanied by precise command to run to produce those results

Adding a table of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility. In some cases, the main result of a paper is a Figure, but that might be more difficult for users to understand without reading the paper. 

You can further help the user understand and contextualize your results by linking back to the full leaderboard that has up-to-date results from other papers. There are [multiple leaderboard services](#results-leaderboards) where this information is stored.  

## 🎉 NeuroBench_LS

### Mask Autoencoder for Lifespan

We provide 2D and 3D MAE

1. [2D MAE](https://github.com/rlu25/NeuroBrench_LS/tree/main/MAE_Model/MAE_2D) - 2D MAE pretrained on slice-based MRI acquisitions.
2. [3D MAE](https://github.com/rlu25/NeuroBrench_LS/tree/main/MAE_Model/MAE_3D) - 3D MAE pretrained on volumetric MRI scans
3. [Google Drive](https://drive.google.com/drive/folders/1JwdIB00tUwrLIUAX7HIbTgUj6c97j6Aa?usp=sharing) - Weights in here

### Baseline

We provide seven SOTA for different tasks

1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers
1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers
1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers
1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers
1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers
1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers
1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers

### Standardized model interfaces

1. [PyTorch Hub](https://pytorch.org/hub/)
2. [Tensorflow Hub](https://www.tensorflow.org/hub)
3. [Hugging Face NLP models](https://huggingface.co/models)
4. [ONNX](https://onnx.ai/)

### Results leaderboards

1. [Papers with Code leaderboards](https://paperswithcode.com/sota) - with 4000+ leaderboards
2. [CodaLab Competitions](https://competitions.codalab.org/) - with 450+ leaderboards
3. [EvalAI](https://eval.ai/) - with 100+ leaderboards
4. [NLP Progress](https://nlpprogress.com/) - with 90+ leaderboards
5. [Collective Knowledge](https://cKnowledge.io/reproduced-results) - with 40+ leaderboards
6. [Weights & Biases - Benchmarks](https://www.wandb.com/benchmarks) - with 9+ leaderboards

### Making project pages

1. [GitHub pages](https://pages.github.com/)
2. [Fastpages](https://github.com/fastai/fastpages)

### Making demos, tutorials, executable papers

1. [Google Colab](https://colab.research.google.com/)
2. [Binder](https://mybinder.org/)
3. [Streamlit](https://github.com/streamlit/streamlit)
4. [CodaLab Worksheets](https://worksheets.codalab.org/)

## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at hello@paperswithcode.com or open an issue on this GitHub repository. 

All contributions welcome! All content in this repository is licensed under the MIT license.
