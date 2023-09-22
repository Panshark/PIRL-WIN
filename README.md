# PIRL-WIN
## Source code for [*Self-Adaptive Driving in Nonstationary Environments through  Conjectural Online Lookahead Adaptation.*](https://github.com/Panshark/COLA/blob/main/icra_colav3.pdf)
A3C model is original implementated by [Palanisamy](https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym) in Chapter 8, the structure of classifer is based on [Rashi Sharma](https://medium.com/swlh/natural-image-classification-using-resnet9-model-6f9dc924cd6d). 

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)

The structure of Neural-SLAM is based on [Neural-SLAM](https://devendrachaplot.github.io/projects/Neural-SLAM).

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [References](#references)
	- [Citing](#citing)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
	- [Contributors](#contributors)
- [License](#license)

### Overview:
The Active Neural SLAM model consists of three modules: a Global Policy, a Local Policy and a Neural SLAM Module. 
As shown below, the Neural-SLAM module predicts a map and agent pose estimate from incoming RGB observations and 
sensor readings. This map and pose are used by a Global policy to output a long-term goal, which is converted to 
a short-term goal using an analytic path planner. A Local Policy is trained to navigate to this short-term goal.

![overview](./docs/overview.png)


## Installing Dependencies
We use earlier versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-api](https://github.com/facebookresearch/habitat-api). The specific commits are mentioned below.

Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9; 
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)

```

Installing habitat-api:
```
git clone https://github.com/facebookresearch/habitat-api.git
cd habitat-api; git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121; 
pip install -e .
```

Install pytorch from https://pytorch.org/ according to your system configuration. The code is tested on pytorch v1.2.0. If you are using conda:
```
conda install pytorch==1.2.0 torchvision cudatoolkit=10.0 -c pytorch #(Linux with GPU)
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch #(Mac OS)
```

## Setup
Clone the repository and install other requirements:
```
git clone --recurse-submodules https://github.com/devendrachaplot/Neural-SLAM
cd Neural-SLAM;
pip install -r requirements.txt
```

The code requires datasets in a `data` folder in the following format (same as habitat-api):
```
Neural-SLAM/
  data/
    scene_datasets/
      gibson/
        Adrian.glb
        Adrian.navmesh
        ...
    datasets/
      pointnav/
        gibson/
          v1/
            train/
            val/
            ...
```
Please download the data using the instructions here: https://github.com/facebookresearch/habitat-api#data

To verify that dependencies are correctly installed and data is setup correctly, run:
```
python main.py -n1 --auto_gpu_config 0 --split val
```


## Usage

### Training:
For training the complete Active Neural SLAM model on the Exploration task:
```
python main.py
```

### Downloading pre-trained models
```
mkdir pretrained_models;
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1UK2hT0GWzoTaVR5lAI6i8o27tqEmYeyY' -O pretrained_models/model_best.global;
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1A1s_HNnbpvdYBUAiw2y1JmmELRLfAJb8' -O pretrained_models/model_best.local;
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1o5OG7DIUKZyvi5stozSqRpAEae1F2BmX' -O pretrained_models/model_best.slam;
```

### For evaluation:
For evaluating the pre-trained models:
```
python main.py --split val --eval 1 --train_global 0 --train_local 0 --train_slam 0 \
--load_global pretrained_models/model_best.global \
--load_local pretrained_models/model_best.local \
--load_slam pretrained_models/model_best.slam 
```

For visualizing the agent observations and predicted map and pose, add `-v 1` as an argument to the above

For more detailed instructions, see [INSTRUCTIONS](./docs/INSTRUCTIONS.md).


## Cite as
>Chaplot, D.S., Gandhi, D., Gupta, S., Gupta, A. and Salakhutdinov, R., 2020. Learning To Explore Using Active Neural SLAM. In International Conference on Learning Representations (ICLR). ([PDF](https://openreview.net/pdf?id=HklXn1BKDH))

### Bibtex:
```
@inproceedings{chaplot2020learning,
  title={Learning To Explore Using Active Neural SLAM},
  author={Chaplot, Devendra Singh and Gandhi, Dhiraj and Gupta, Saurabh and Gupta, Abhinav and Salakhutdinov, Ruslan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## Acknowledgements
This repository uses Habitat API (https://github.com/facebookresearch/habitat-api) and parts of the code from the API.
The implementation of PPO is borrowed from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/.
We thank Guillaume Lample for discussions and coding during initial stages of this project.
