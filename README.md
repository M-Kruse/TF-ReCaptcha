# TF-ReCaptcha

This is a project to solve the ReCaptcha puzzle using Selenium and Tensorflow Object Detection API or maybe another object detection framework.

You can interact with individual ReCaptcha puzzle elements through the RecaptchaElement class. Currently this supports clicking the element and rendering the image of the element.

I noticed a couple things about recaptcha so far:

* You can sometimes force recaptcha to fall back to the easiest type, the fire hydrant, if you repeatedly fail the other tests.
* Sometimes the images look like they have been fuzzed to counter NN object detection. I've heard of techniques to fool object detection while still making the image human readable and I think this is being used here

Farming mode won't attempt to solve it, it will just collect the image(s) and fail it intentionally. The point is to collect as many images as possible for training.

Here is the command to run from my docker image

`docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v $PWD:/tmp  -w /tmp  -u tf-user:docker  -it  --rm tf-obj-detect-api python tf_obj_api.py`

# Running from TF Object Detection API Docker build - https://github.com/M-Kruse/docker-tf-obj-detection-api

```
scooty@ScootysUbuntu:~/Code/TF-ReCaptcha$ PYTHONUNBUFFERED=0 docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v $PWD:/tmp  -w /tmp  -u tf-user:docker  -it  --rm tf-obj-detect-api python tf_obj_api.py
2020-05-21 05:26:31.380859: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 05:26:31.403795: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 3999980000 Hz
2020-05-21 05:26:31.404391: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f81fc000b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 05:26:31.404417: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-05-21 05:26:31.408233: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-05-21 05:26:31.622400: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-05-21 05:26:31.622788: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4061650 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-05-21 05:26:31.622805: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2070, Compute Capability 7.5

...
...

2020-05-21 05:26:32.994690: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4787 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2070, pci bus id: 0000:01:00.0, compute capability: 7.5)
[<tf.Tensor 'image_tensor:0' shape=(None, None, None, 3) dtype=uint8>]
2020-05-21 05:26:38.721427: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-21 05:26:39.632293: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
models/research/object_detection/test_images/image1.jpg
['dog']
models/research/object_detection/test_images/image2.jpg
['person', 'kite']
scooty@ScootysUbuntu:~/Code/TF-ReCaptcha$ 
```