# DisguisOR: Holistic Face Anonymization for the Operating Room
<a href="https://arxiv.org/abs/2307.14241" target="_blank">[arXiv]</a> 
<a href="https://wngtn.github.io/disguisor" target="_blank">[project page]</a> 

<p align="center">
  <img src="assets/main_figure.jpg">
</p>

This repository contains the code for our paper [DisguisOR: Holistic Face Anonymization for the Operating Room](https://arxiv.org/abs/2307.14241) by Lennart Bastian, Tony Danjun Wang, Tobias Czempiel, Benjamin Busam, Nassir Navab.

It includes:
- [3D Human Pose Key-Point Detection](#3d-human-pose-key-point-detection)
- [3D Human Mesh Estimation](#3d-human-pose-key-point-detection)
- [Anonymization](#anonymization)

## Setup

### Python Environment
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

```
conda create --name disguisor python=3.9
```

### Dataset
We publish 75 frames of the *hard* scenario used in our paper for demonstration purposes. Each frame contains the RGB image,
point cloud and depth map for each camera.

To download the demo dataset, please first fill out [this Google form](https://forms.gle/7UUg788xmj3sB18PA). 
Upon submission you should receive an email with the download link and password for unzipping. Download the dataset and put it into `data/`.
Your file structure should be like this:

```
data
└── hard
    ├── cn01 # contains RGB/Point Cloud/Depth Maps for camera 1
    ├── cn02 
    ├── cn03
    ├── cn04
    └── hard_2d_kps.pkl # contains 2D human pose key-points for each camera per frame
```

Note that one of four individuals in this data is pre-anonymized manually by blurring the eyes.

## 3D Human Pose Key-Point Detection

In this paper, we use [VoxelPose](https://github.com/microsoft/voxelpose-pytorch) to estimate 3D human pose key-points
in the scene.

[VoxelPose](https://github.com/microsoft/voxelpose-pytorch) first requires 2D human pose key-points as input. We provide
pre-computed 2D human pose key-points for each camera per frame in `data/hard/hard_2d_kps.pkl` [(see dataset)](#dataset).
These human pose key-points were detected with [DEKR](https://github.com/HRNet/DEKR) using the `pose_hrnet_w48` model 
trained on COCO.

You can also skip this section if you are only interested in anonymization by downloading the human mesh models as explained [here](#generate-3d-human-mesh-models).

### Setup

First, change directory to `external/VoxelPose`:
```
cd external/VoxelPose
```

Install all necessary libraries:
```
pip install -r requirements.txt
```

### Training VoxelPose

Download a set of human poses from [here](https://nextcloud.in.tum.de/index.php/s/aSyebAJACoap4eZ) that is used to create
synthetic data for training. Save it to `data/panoptic_training_pose.pkl`. Note that the current directory is `external/VoxelPose`.

Run this to train a model from scratch using synthetic data (generated in `lib/dataset/disguisor_synthetic`):
```
python run/train_3d.py --cfg configs/hard.yaml 
```
The trained model can then be found in `output/disguisor_synthetic/multi_person_posenet_50/hard`.

### Inference: Creating 3D Human Pose Key-Points

We provide a [pre-trained VoxelPose model](https://nextcloud.in.tum.de/index.php/s/Ar5PkqfiRqXKLYm) for inference.
Save it under `pre_trained/hard.pth.tar`.

Run the following to create 3D human pose key-points and visualize them in 2D by projecting them into the cameras (-v flag):
```
python run/create_3d_kps.py --cfg configs/hard.yaml --pretrained_model pre_trained/hard.pth.tar -v
```

The output is automatically saved in the `external/EasyMocap/data/hard/output/keypoints3d` folder for the next step - 3D 
human mesh generation.

## 3D Human Mesh Generation

In this paper, we use [EasyMoCap](https://github.com/zju3dv/EasyMocap) to generate 3D human meshes from 3D key-points.

### Setup

First, change to the directory `external/EasyMocap`

```
cd external/EasyMocap
```

Now, similarly, install all necessary libraries and build the package.
```
pip install -r requirements.txt
python setup.py develop
```

Download the [SMPL models](https://smpl.is.tue.mpg.de) and place them like this:

```
data
└── smplx
    └── smpl
        └── smpl
            ├── SMPL_FEMALE.pkl
            ├── SMPL_MALE.pkl
            └── SMPL_NEUTRAL.pkl
```

### Generate 3D Human Mesh Models

You can skip these steps and directly download the SMPL meshes [here](https://nextcloud.in.tum.de/index.php/s/DHoXbQABseBSZSb). Extract the folder and place it under `input/hard/smpl_meshes`. Note that this directory is assuming `.` as the current directory.

We first set the directory of our 3D human pose key-points generated in [this step](#3d-human-pose-key-point-detection).
```
data=data/hard/
``` 

Then, we track the human pose in each frame. This step will track and interpolate missing frames (you might need to install a different numpy version for this step: `pip install numpy==1.23`):
```
python apps/demo/auto_track.py ${data}/output ${data}/output-track --track3d
```

Then, we can fit SMPL model to the tracked key-points:
```
python apps/demo/smpl_from_keypoints.py ${data} --skel ${data}/output-track/keypoints3d --out ${data}/output-track/smpl --verbose --opts smooth_poses 1e1 --body COCO
```

At last, we can convert the EasyMocap `.json` output format into 3D `.obj` file format:
```
python apps/postprocess/write_meshes.py
```

This automatically saves the 3D `.obj` files into `../../input/hard/smpl_meshes`.

## Anonymization

### Setup

Make sure you are in the root folder of this repository.

Install all necessary libraries:

```
pip install -r requirements.txt
```

⚠️ You additionally need the [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library to render the faces, follow [these steps](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for installation.


### Anonymize the images

Run this command, which performs the last step of anonymization. It uses the 3D human meshes generated in the [previous step](#3d-human-mesh-generation), extracts their head and aligns them in 3D with the point cloud. Then the face segment is extracted and reprojected back in 2D, replacing the faces of each individual.
```
python run/main.py --config configs/disguisor_hard.yaml 
```

The 75 anonymized 2D RGB frames (4 frames each) are then saved under `output/hard/anonymized_images`.

Additionally, if the flag (`--add_bboxes`) is raised, the anonymized 2D RGB frames and the detected bounding box (closest rectangle around the rendered mesh) are saved under `output/hard/anonymized_bbox_images`.

Note that due to the probabilistic nature of the point cloud downsampling and registration method ([Filterreg](https://github.com/neka-nat/probreg)), the head alignment is inadequate sometimes if the point cloud contains inaccuracies.

### Visualization

You can use the visualization script in `scripts/visualize_anonymization.py` to visualize the anonymized images by running:
```
python scripts/visualize_anonymization.py
```
This will create an mp4 video in `output/hard/video.mp4`

# Citation

If you find this work useful for your research or project, please consider citing our paper. 

```bibtex
@article{bastian2023disguisor,
    author    = {Bastian, Lennart and 
                Wang, Tony D. and 
                Czempiel, Tobias and 
                Busam, Benjamin and 
                Navab, Nassir},
    title     = {DisguisOR: Holistic Face Anonymization in the Operating Room},
    journal   = {IPCAI},
    year      = {2023},
}
```

