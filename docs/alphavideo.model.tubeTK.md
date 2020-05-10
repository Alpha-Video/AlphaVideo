### Introduction
TubeTK is an end-to-end one training stage model for video multi-object tracking.
This is the official implementation of paper 
["TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model"](https://bopang1996.github.io/posts/2020/04/tubeTKpaper/).
The detailed intact training and inference scripts can be found [here](https://github.com/BoPang1996/TubeTK).
### API
>*FUNCTION* alphavideo.model.tubeTK(num_class=1, pretrain=True)

* Build the tubeTK model for multi-object tracking.
* Parameters:
  * ```num_class (int)```: Number of object categories the model tracks. 
  At present, ```num_class=1``` is usually used for pedestrian or car tracking.
  We only provide pretrained model for ```num_class=1```. By default ```num_class=1```.
  
  * ```pretrain (bool)```: Whether to load weight pretrained on [MOT16](https://motchallenge.net/data/MOT16/). 
  We only provide pretrained model for ```num_class=1```.
  By default, ```pretrain=True'''.
  
* Input:
  * ```img (tensor)```: Input frames of the target video.
  Its shape is (<img src="http://chart.googleapis.com/chart?cht=tx&chl=N, C_{in}, T, H, W" style="border:none;">). By default, <img src="http://chart.googleapis.com/chart?cht=tx&chl=T=8, H=896, W=1152" style="border:none;">.
  
  * ```img_meta (list of dic)```: Meta data for input frames.
    It is a list of dic. Each dic is corresponding for a video in the batch.
    The shape of dic is:
      ```
      {'img_shape': [8, 1080, 1920],
       'value_range': 1,
       'pad_percent': [1,1]}
      ```
      where ```img_shape``` indicates the original shape of the input clip before transformation like ```resize``` or ```padding```. 
      It is used for mapping the predicted coordinates to original space. 
      ```value_range``` is the value the model used to present the coordinate. 
      For example, if ```value_range=2```, ```(2, 2)``` will be the coordinate of bottom right corner.
      ```pad_percent``` indicated the padding percent of the input frames. 
      For example, if an image with original shape of (80, 100) is padding to (100, 100) for input, the ```pad_percent``` should be ```[1, 0.8]```.
    
  * ```gt_tubes (list of tensor)```: Only needed when training. 
  It is a list of tensor. Each tensor is corresponding for a video in the batch.
  The shape of tensor is ```n, 15```, representing ```n``` Btubes which is expressed by 15 coordinates.
  For example, if a Btube's <img src="http://chart.googleapis.com/chart?cht=tx&chl=B_s" style="border:none;"> is <img src="http://chart.googleapis.com/chart?cht=tx&chl=(t_s, x^0_s, y^0_s, x^1_s, y^1_s)" style="border:none;">, <img src="http://chart.googleapis.com/chart?cht=tx&chl=B_m" style="border:none;"> is <img src="http://chart.googleapis.com/chart?cht=tx&chl=(t_m, x^0_m, y^0_m, x^1_m, y^1_m)" style="border:none;">, and
  <img src="http://chart.googleapis.com/chart?cht=tx&chl=B_e" style="border:none;"> is <img src="http://chart.googleapis.com/chart?cht=tx&chl=(t_e, x^0_e, y^0_e, x^1_e, y^1_e)" style="border:none;">, then the input coordinates should be 
  <img src="http://chart.googleapis.com/chart?cht=tx&chl=(x^0_m, y^0_m, x^1_m, y^1_m, t_m, t_e - t_m, x^0_e - x^0_m, y^0_e - y^0_m, x^1_e - x^1_m, y^1_e - y^1_m, t_s - t_m, x^0_s - x^0_m, y^0_s - y^0_m, x^1_s - x^1_m, y^1_s - y^1_m)" style="border:none;">.
  
  * ```gt_labels (list of tensor)```: Only needed when training.
  It is a list of tensor. Each tensor is corresponding for a video in the batch.
  The shape of tensor is ```n, num_class```, representing ```n``` one-hot labels.
  
  * ```return_loss (bool)```: A flag to control whether to train the model and return the loss (True) or conduct inference process and return Btube list (False).
  
* Output:
  * When ```return_loss=True```: The output is a dic containing multiple loss:
    ```python
      dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg,
            loss_centerness=loss_centerness,
            mid_iou_loss=mid_iou_loss)
    ```
  * When ```return_loss=False```: The output is a list of results.
  Each element in the list is the results of one video in the batch.
  The results is also a list: ```[tubes, labels, others]```. 
  ```tubes``` is a list of Btubes with shape ```[n ,15]``` just as the input ```gt_tubes```.
  ```labels``` is a list of lables with shape ```[n, num_class]``` just as the input ```gt_labels```.
  ```others``` is some intermediate results. For details, please see detailed train and evaluation scripts [here](https://github.com/BoPang1996/TubeTK).
  

* Example:
```python
# model
model = alphavideo.model.tubeTK(pretrain=True)
print(model)

# input
images = torch.zeros((1, 3, 8, 896, 1152))
image_meta = [{'img_shape': [8, 1080, 1920],
               'value_range': 1,
               'pad_percent': [1, 1]}]
gt_tubes = [torch.tensor([[0, 0, 0.1, 0.1, 3, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0]])]
gt_labels = [torch.ones((1, 1))]
results = model(images, image_meta, return_loss=False, gt_tubes=gt_tubes, gt_labels=gt_labels)
print(results)
```

### Citation
```
@inproceedings{pang2020tubeTK,
  title={TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model},
  author={Pang, Bo and Li, Yizhuo and Zhang, Yifan and Li, Muchen and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

