## Introduction
AlphaVideo is an open-sourced video understanding toolbox based on [PyTorch](https://pytorch.org/) covering multi-object tracking and action detection.
In AlphaVideo, we released the first one-stage multi-object tracking (MOT) system **TubeTK** that can achieve 66.9 MOTA on [MOT-16](https://motchallenge.net/results/MOT16) dataset and 63 MOTA on [MOT-17](https://motchallenge.net/results/MOT17) dataset.
For action detection, we released an efficient model **AlphAction**, which is the first open-source project that achieves 30+ mAP (32.4 mAP) with single model on [AVA](https://research.google.com/ava/) dataset.

## Quick Start
### pip
Run this command:
```shell
pip install alphavideo
```

### from source
Clone repository from github:
```bash
git clone https://github.com/Alpha-Video/AlphaVideo.git alphaVideo
cd alphaVideo
```

Setup and install AlphaVideo:
```bash
pip install .
```

## Features & Capabilities 
* #### Multi-Object Tracking
  For this task, we provide the [TubeTK](https://github.com/BoPang1996/TubeTK) model which is the official implementation of paper 
  "TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model (CVPR2020, **oral**)." 
  Detailed training and testing script on [MOT-Challenge](https://motchallenge.net/) datasets can be found [here](https://github.com/BoPang1996/TubeTK).
  
  <img src="https://github.com/BoPang1996/TubeTK/raw/master/assets/demo.gif" width = "600" align=center />
    
    * Accurate end-to-end multi-object tracking.
    * Do not need any ready-made image-level object deteaction models.
    * Pre-trained model for pedestrian tracking. 
    * Input: Frame list; video.
    * Output: Videos decorated by colored bounding-box; Btube lists.
    * For details usages, see our [docs](https://github.com/Alpha-Video/AlphaVideo/wiki).

* #### Action recognition

  For this task, we provide the [AlphAction](https://github.com/MVIG-SJTU/AlphAction) model as an implementation of paper ["Asynchronous Interaction Aggregation for Action Detection"](https://arxiv.org/abs/2004.07485). This paper is recently accepted by **ECCV 2020**!
  
  <img src="https://github.com/MVIG-SJTU/AlphAction/raw/master/gifs/demo2.gif" width = "600" align=center />
    
    * Accurate and efficient action detection.
    * Pre-trained model for 80 atomic action categories defined in [AVA](https://research.google.com/ava/).
    * Input: Video; camera.
    * Output: Videos decorated by human boxes, attached with corresponding action predictions.
    * For details usages, see our [docs](https://github.com/Alpha-Video/AlphaVideo/wiki).

## Paper and Citations
```
@inproceedings{pang2020tubeTK,
  title={TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model},
  author={Pang, Bo and Li, Yizhuo and Zhang, Yifan and Li, Muchen and Lu, Cewu}
  booktitle={CVPR},
  year={2020}
}

@inproceedings{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2020}
}
```

## Maintainers
This project is open-sourced and maintained by Machine Vision and Intelligence Group ([MVIG](http://mvig.sjtu.edu.cn)) in Shanghai Jiao Tong University.

