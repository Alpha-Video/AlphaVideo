### Introduction
AlphAction aims to detect the actions of multiple persons in videos.
This is the embedded implementation of paper 
["Asynchronous Interaction Aggregation for Action Detection"](https://arxiv.org/abs/2004.07485).
The standalone implementation could be found at [here](https://github.com/MVIG-SJTU/AlphAction).

### API
>*FUNCTION* alphavideo.model.alphaction_res50(num_class=80, pretrain=True)

>*FUNCTION* alphavideo.model.alphaction_res101(num_class=80, pretrain=True)

* Build the ResNet-50 (or ResNet-101) AlphAction model for spatio-temporal action detection.
* Parameters:
  * ```num_class (int)```: Number of ation categories the model detects.
  By default ```num_class=80```.
  
  * ```pretrain (bool)```: Whether to load pretrained weight. 
  We only provide pretrained model for ```num_class=80```, which is pretrained on [AVA](https://research.google.com/ava/) dataset.
  For the list of action categories, you
  can refer to the [official website](https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt).
  By default, ```pretrain=True```.
  
* Input:
  * ```video_clip (tensor)```: Input frames of the target video.
  Its shape is ![(N,C_{in},T,H,W)](https://render.githubusercontent.com/render/math?math=(N%2CC_%7Bin%7D%2CT%2CH%2CW)). Make sure ![T=32](https://render.githubusercontent.com/render/math?math=T%3D32)  and ![H=256](https://render.githubusercontent.com/render/math?math=H%3D256) is recommended.

  * ```boxes (list of Tensor)```: The bounding boxes of target persons in the middle frame of each video clip. The length of the list should be ![N](https://render.githubusercontent.com/render/math?math=N). Each Tensor in the list has the shape ![(num_{person}, 4)](https://render.githubusercontent.com/render/math?math=(num_%7Bperson%7D%2C%204)). Each boxes are in the format ![(x_1, y_1, x_2, y_2)](https://render.githubusercontent.com/render/math?math=(x_1%2C%20y_1%2C%20x_2%2C%20y_2)), where ![(x_1, y_1)](https://render.githubusercontent.com/render/math?math=(x_1%2C%20y_1)) and ![(x_2, y_2)](https://render.githubusercontent.com/render/math?math=(x_2%2C%20y_2)) are the absolute coordinates of the top-left corner and the bottom-right corner. If the element in the list is a list, it will be converted to a Tensor implicitly.
  
  * ```video_metas (list of dict)```: Meta data for input video clips.
  It is a list of dictionary objects. Each dictionary is for a specific clip in the batch.
  The shape of each dictionary is:

    ```python
    {
      'movie_id': "videoname",
      'timestamp': 1
    }
    ```

    where ```movie_id``` is the string used to identify video clips from different long videos and ```timestamp``` is an integer which denotes the index of each clip in its corresponding long video. Both two values are provided for composing memory features. For a target clip with timestamp ```t```, the memory features are gathered from ```t-30``` to ```t+30``` (if available) .
  
  * ```objects (list of Tensor)```: This input argument is optional. If given, it should be the bounding boxes of objects detected in the middle frame of each video clip. These boxes are used to generate object features. The detailed format is just the same as the argument ```boxes```.

  
* Output:
  The output is a list of ```BoxList``` objects. Each object in this list is the detection result of one target clip. The attribute ```BoxList.bbox``` is the bounding boxes Tensor in the same format as the input argument ```boxes```. For ![num_{person}](https://render.githubusercontent.com/render/math?math=num_%7Bperson%7D) target persons and ![num_{class}](https://render.githubusercontent.com/render/math?math=num_%7Bclass%7D) action categories, the method function ```BoxList.get_field("scores")``` will return the scores of each action categories as a ![(num_{person}, num_{class})](https://render.githubusercontent.com/render/math?math=(num_%7Bperson%7D%2C%20num_%7Bclass%7D)) Tensor. 
  

* Example:
  * Online case: For the online case where future memory features is not available, the usage is as simple as follows.

    ```python
    import torch

    # model
    model = alphavideo.model.alphaction_res101(pretrain=True).cuda()

    # online use, no future memory feature
    with torch.no_grad():
        for t in range(10):
            videos = torch.zeros((1, 3, 32, 256, 256)).cuda()
            video_metas = [dict(movie_id="tmp", timestamp=t)]
            box = [[[1,1,10,10],[20,30,80,100]]]
            objects = [[[50,50,60,60]]]
            results = model(videos, box, video_metas, objects)
            print(results[0].get_field("scores"))
    
    # clear memory feature
    model.empty_pool()
    ```

  * Offline case: For the offline case where future memory features is available. You should first get the full memory features of different timestamps by the function ```model.store_feature``` and then get the output of each clip by ```model.get_prediction```. 

    ```python
    import torch

    # model
    model = alphavideo.model.alphaction_res101(pretrain=True).cuda()

    # offline use, use future memory feature
    with torch.no_grad():
        # extract features for each clip in the same long video.
        for t in range(10):
            videos = torch.zeros((1, 3, 32, 256, 256)).cuda()
            video_metas = [dict(movie_id="tmp", timestamp=t)]
            box = [[[1,1,10,10],[20,30,80,100]]]
            objects = [[[50, 50, 60, 60]]]
            model.store_feature(videos, box, video_metas, objects)
        device = videos.device

        # get the detection result of each clip.
        for t in range(10):
            video_metas = [dict(movie_id="tmp", timestamp=t)]
            results = model.get_prediction(video_metas=video_metas, device=device)
            print(results[0].get_field("scores"))
    
    # clear memory feature
    model.empty_pool()
    ```

### Citation
```
@article{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  journal={arXiv preprint arXiv:2004.07485},
  year={2020}
}
```