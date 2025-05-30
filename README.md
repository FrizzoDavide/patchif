# MoViAD: Modular Visual Anomaly Detection
![image](https://github.com/user-attachments/assets/44879986-5282-4c86-a3b3-0aaa61a7103f)

## How to Install

Inside the main repository directory, run the following command:

Editable mode (if you need to work on the code):

```bash
pip install -e ./
```

Fixed mode (if you just want to use the code):

```bash
pip install ./
```

## Execution example

Inside the <code>/main_scripts</code> directory are present some execution scripts for training and testing the AD models. 

For example, for training patchcore: 

```bash
python main_scripts/main_patchcore.py --mode train --dataset_path /home/datasets/mvtec --category pill --backbone mobilenet_v2 --ad_layers features.4 features.7 features.10 --device cuda:0 --save_path ./patch.pt 
```

For every main script all its parameters are documented. 

## Project Objective

Apply TinyML architectures to existing AD models to evaluate the prediction performance when reducing the computational cost of the overall model.

Achieve computer vision anomaly detection in devices with limited hardware resources such as limited computational power and memory.
Specifically, we are particularly interested in _pixel-level_ anomaly detection, so the evaluation of the model will take into account its segmentation capabilities.

## Methods and Architectures

Several AD methods will be tested against different backbones, where the backbone is usually a pre-trained feature extraction model for computer vision tasks.

The models will be evaluated using the industry standard anomaly detection datasets.

**Machine learning paradigms**

- Unsupervised Learning: the model is trained only on a part of the dataset which does not have anomalies. This is desirable when time or economic constraints prevent the use of a labeled dataset.
  - Self Supervised Learning
    - Contrastive learning: unlabeled data points are juxtaposed against each other to teach a model which points are similar and which are different.

**AD Models**

- PatchCore: [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html), [code](https://github.com/amazon-science/patchcore-inspection)
- CFA [paper](https://ieeexplore.ieee.org/abstract/document/9839549), [code](https://github.com/sungwool/CFA_for_anomaly_localization)
- Student-Teacher Feature Pyramid: [paper](https://arxiv.org/abs/2103.04257), [code](https://github.com/gdwang08/STFPM)
- PaDiM: [paper with code](https://paperswithcode.com/paper/padim-a-patch-distribution-modeling-framework)
- PatchSVDD: [papers with code](https://paperswithcode.com/paper/patch-svdd-patch-level-svdd-for-anomaly)
- PatchIF → under development

**Feature Extraction Backbones**

- MobileNet: [link](https://paperswithcode.com/paper/mobilenets-efficient-convolutional-neural)
- PhiNet: [link](https://paperswithcode.com/paper/phinets-a-scalable-backbone-for-low-power-ai)
- MicroNet: [link](https://paperswithcode.com/paper/micronet-improving-image-recognition-with)
- MCUNet: [link](https://paperswithcode.com/paper/mcunet-tiny-deep-learning-on-iot-devices)
- XiNet: [link](https://paperswithcode.com/paper/xinet-efficient-neural-networks-for-tinyml)

**Datasets**

- MVTecAD: [link](https://paperswithcode.com/dataset/mvtecad)
- MVTec LOCO AD: [link](https://paperswithcode.com/dataset/mvtec-loco-ad)
- VisA: [link](https://paperswithcode.com/dataset/visa)

## Contribute

If you want to contribute to the repository, follow the present code structure: 
- inside the <code>/models/model_name</code> directory put the code for possible new anomaly detection models
- inside the <code>/trainers</code> directory put the code for training an anomaly detection model
- inside the <code>/datasets</code> directory put the code for possible new anomaly detection datasets that must be used

Every contribution must be open with a pull request. 
