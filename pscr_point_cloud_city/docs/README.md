
<p align="center">
<img src="https://raw.githubusercontent.com/alexdimopoulos/Open3D-ML_Point_Cloud_City/main/data/nist_pscr_logo.png" 
width="420" />
<span style="font-size: 220%"><b>Point Cloud City on</b></span>
<br>  
<img src="https://raw.githubusercontent.com/isl-org/Open3D/main/docs/_static/open3d_logo_horizontal.png" width="320" />
<span style="font-size: 220%"><b>ML</b></span>
</p>

![Ubuntu CI](https://github.com/isl-org/Open3D-ML/workflows/Ubuntu%20CI/badge.svg)
![Style check](https://github.com/isl-org/Open3D-ML/workflows/Style%20check/badge.svg)
![PyTorch badge](https://img.shields.io/badge/PyTorch-supported-brightgreen?style=flat&logo=pytorch)
![TensorFlow badge](https://img.shields.io/badge/TensorFlow-supported-brightgreen?style=flat&logo=tensorflow)

[**Installation**](#installation) | [**Get started**](#getting-started-with-point-cloud-city-and-open3d-ml) | [**Structure**](#repository-structure) | [**Tasks & Algorithms**](#tasks-and-algorithms) | [**Model Zoo**](model_zoo.md) | [**Datasets**](#datasets) | [**How-tos**](#how-tos) | [**Contribute**](#contribute)


[Point Cloud City](https://www.nist.gov/ctl/pscr/funding-opportunities/past-funding-opportunities/psiap-point-cloud-city) was developed during the 2018 NIST Public Safety Innovation Accelerator Program - Point Cloud City NOFO awardees generated an extensive catalog of annotated 3D indoor point clouds that can be used by industry, academia, and government to advance research and development in the areas of indoor mapping, localization and navigation for public safety, as well as to demonstrate the potential value of ubiquitous indoor positioning and location-based information. These pioneering U.S. state and local governments will create a model ‘Point Cloud City’ and also participate in the NIST Global Cities Team Challenge initiative as the lead for an Action Cluster. This repository extends Open3D-ML to be imlemented using the Point Cloud City datasets and features the processing code, dataset integration, and machine learning model configuration files.

Open3D-ML is an extension of Open3D for 3D machine learning tasks.
It builds on top of the Open3D core library and extends it with machine learning
tools for 3D data processing. This repo focuses on applications such as semantic
point cloud segmentation and provides pretrained models that can be applied to
common tasks as well as pipelines for training.

Open3D-ML-PointCloudCity works with **TensorFlow** and **PyTorch** to integrate easily into
existing projects and also provides general functionality independent of
ML frameworks such as data visualization.


<!-- ![Visualizer GIF](docs/images/getting_started_ml_visualizer.gif) -->
<img src="https://raw.githubusercontent.com/alexdimopoulos/Open3D-ML_Point_Cloud_City/master/data/pcc_indoor_viz.png"/>



## Results

### Ground Truth - Point Cloud City 
---
<p float="left">
  <img src="https://raw.githubusercontent.com/alexdimopoulos/Open3D-ML_Point_Cloud_City/master/data/gt_stairs_70.png" width="400" height="386" /> 
  <img src="https://raw.githubusercontent.com/alexdimopoulos/Open3D-ML_Point_Cloud_City/master/data/enfield_student_union_kpconv_gt70.png" width="400" />
</p>

### KPCONV Semantic Segmentation Results - PCC-SKITTI
---
<p float="left">
  <img src="https://raw.githubusercontent.com/alexdimopoulos/Open3D-ML_Point_Cloud_City/master/data/enfield_student_union_kpconv_results70.png" width="400" /> 
  <img src="https://raw.githubusercontent.com/alexdimopoulos/Open3D-ML_Point_Cloud_City/master/data/results_stair.png" width="400" height="389" />
</p>

## Repository structure
The core part of Open3D-ML lives in the `ml3d` subfolder, which is integrated
into Open3D in the `ml` namespace. In addition to the core part, the directories
`examples` and `scripts` provide supporting scripts for getting started with
setting up a training pipeline or running a network on a dataset.

```
├─ pscr_point_cloud_city  # Documentation and processing code for PCC O3D-ML 
     ├─ docs              # Directory for dataset details
     ├─ scripts           # Dataset processing code
├─ docs                   # Markdown and rst files for documentation
├─ examples               # Place for example scripts and notebooks
├─ ml3d                   # Package root dir that is integrated in open3d
     ├─ configs           # Model configuration files
     ├─ datasets          # Generic dataset code; will be integratede as open3d.ml.{tf,torch}.datasets
     ├─ metrics           # Metrics available for evaluating ML models
     ├─ utils             # Framework independent utilities; available as open3d.ml.{tf,torch}.utils
     ├─ vis               # ML specific visualization functions
     ├─ tf                # Directory for TensorFlow specific code. same structure as ml3d/torch.
     │                    # This will be available as open3d.ml.tf
     ├─ torch             # Directory for PyTorch specific code; available as open3d.ml.torch
          ├─ dataloaders  # Framework specific dataset code, e.g. wrappers that can make use of the
          │               # generic dataset code.
          ├─ models       # Code for models
          ├─ modules      # Smaller modules, e.g., metrics and losses
          ├─ pipelines    # Pipelines for tasks like semantic segmentation
          ├─ utils        # Utilities for <>
├─ scripts                # Demo scripts for training and dataset download scripts
```


## Tasks and Algorithms

### Semantic Segmentation

For the task of semantic segmentation, we measure the performance of different methods using the mean intersection-over-union (mIoU) over all classes.
The table shows the available models and datasets for the segmentation task and the respective scores. Each score links to the respective weight file.

This table displays results using the model [KPCONV](https://arxiv.org/abs/1904.08889) and was run using PyTorch.


| Dataset | Metric - Mean | Stairway | Windows | Roof Access | Fire Sprinkler | Gas Shutoff|
|--------------------|---------------|----------- |-------|--------------|-------------|---------|
| Enfield | IoU | 37.8 |  50.1 |  23.7 | 20.1 | 68.6 
|         |  F1  | 54.9 |  65.1 |  66.8 | 34.1 | 81.4 
| Memphis | IoU | 6.7  |  97.6 |  0.00 | 0.05 | 0.00 
|         |  F1  | 13.1 |  98.87 |  0.00 | 0.10 | 0.00 
| PCC_SKITTI |  IoU | 28.9 |  92.7 |  42.8 | 77.8 | 76.7 
|            |  F1   | 28.9 |  92.7 |  42.8 | 25.3 | 77.8 

(*) Using weights trained on the Point Cloud City with mean calculated from test datasets. PCC_SKITTI is the combination of the Enfield and Memphis datasets with unified labels and formatted to replicate [Semantic KITTI's](http://www.semantic-kitti.org/) structure.








