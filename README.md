# General Multi-Class Classifier
##### A classifier that uses Inception to classify labeled images that can be used as a custom trainable object detection and recognition service from applications
&nbsp;
[![Tensorflow](https://seedroid.com/img/post/icons/128/cc.nextlabs.tensorflow.jpg)](https://nodesource.com/products/nsolid)      [![CUDA](http://www.channelpronetwork.com/sites/default/files/styles/large/public/thumbnails/news//nvidia-cuda0.jpg?itok=TgfuEHhw)](https://nodesource.com/products/nsolid)


## Inspiration
This project is greatly inspired from the ImageNet and ResNet.

### Technology

Oversight uses a number of open source projects to work properly:

* [Tensorflow] - A google open-source ML framework
* [Python] - awesome language we love

### Neural Network Diagram for `Inceptionv3`
[![Architecture](https://raw.githubusercontent.com/pourabkarchaudhuri/general-multiclass-classifier/master/neural_schema.png)](https://nodesource.com/products/nsolid)

### Environment Setup

##### This was built on Windows 10.

These were the pre-requisities :

##### NVIDIA CUDA Toolkit
* [CUDA] - parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). Download and Install all the patches. During install, choose Custom and uncheck the Visual Studio Integration checkbox.

##### Download cuDNN
* [cuDNN] - The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. Create a NVIDIA developer account to download.

##### Set Path :
Add the following paths,
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
&nbsp;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\extras\CUPTI\libx64

##### Install [Anaconda](https://www.anaconda.com/download/) with 3.6 x64

```sh
$ conda update conda
```

##### Run package installer

```sh
$ pip install -r requirements.txt
```



## Instructions

### Create `training_dataset` folder and maintain this structure of image data
```
/
--- /training_dataset
|    |
|    --- /circle
|    |    circle1.jpg
|    |    circle_small_red.png
|    |    ...
|    |
|    --- /square
|         square.jpg
|         square3.jpg
|         ...
```
### To Train : Run `train.sh` or
```
$ python retrain.py \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=tf_files/training_summaries/basic \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --image_dir=training_dataset
```
### To Classify : Run 
```
$ python classify.py testimage.jpg
```
### Todos

 - Optimize Further to increase speed
 - Implement Docker and Jenkins based deployment

License
----

Public

   [Tensorflow]: <https://www.tensorflow.org/>
   [Python]: <https://www.python.org/>
   [Google's FaceNet]: <https://arxiv.org/abs/1503.03832>
   [Anaconda]: <https://www.anaconda.com/download/>
   [CUDA]: <https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal>
   [cuDNN]: <https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7>

  
