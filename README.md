# TF-ReCaptcha

This is a project to solve the ReCaptcha puzzle using Selenium and Tensorflow Object Detection API or maybe another object detection framework.

You can interact with individual ReCaptcha puzzle elements through the RecaptchaElement class. Currently this supports clicking the element and rendering the image of the element.

The 3x3 seems easier to solve, so that is the first goal. 

I noticed some things about recaptcha so far:

* You can sometimes force recaptcha to fall back to the easiest type, the fire hydrant, if you repeatedly fail the other tests.
* Sometimes the images look like they have been fuzzed to counter NN object detection. I've heard of techniques to fool object detection while still making the image identifiable by a human and I think this is being used here
* The number of anti-NN object detection images seems to increase the more you access the recaptcha from a single IP.
* You can access the 3x3 as a single image, otherwise the grid elements have single image sources.
* The 4x4 is a larger single image also with a grid overlay
* I haven't seen any 4x4 puzzle type where the image looks like its been modified against object detection.

Harvesting mode won't attempt to solve it, it will just collect the image(s) and fail it intentionally. The point is to collect as many images as possible for training.

Here is an example usage of interacting with the puzzle elements, it is not trained for crosswalks but you should get the idea.

https://streamable.com/8rvkr4

```
Python 3.8.2 (default, Apr 27 2020, 15:53:34) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from tf_recaptcha import TFRecaptcha
>>> TFR = TFRecaptcha()
>>> TFR.init_tf()
2020-05-22 15:14:14.324338: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 15:14:14.361854: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 3999980000 Hz
...
...
2020-05-22 15:14:14.367631: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:310] kernel version seems to match DSO: 440.64.0
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
>>> TFR.init_browser()
True
>>> TFR.open_recaptcha()
[INFO] Attempting to bypass 4x4 puzzle type
>>> TFR.generate_recaptcha_classes()
>>> TFR.recaptchas
[<tf_recaptcha.RecaptchaElement object at 0x7f46eef5e850>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8d9d0>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8db80>, <tf_recaptcha.RecaptchaElement object at 0x7f4676370df0>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8d550>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8dca0>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8d5e0>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8d3a0>, <tf_recaptcha.RecaptchaElement object at 0x7f45e3f8d4c0>]
>>> TFR.recaptcha[0].render_img()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'TFRecaptcha' object has no attribute 'recaptcha'
>>> TFR.recaptchas[0].render_img()
>>> TFR.recaptchas[1].render_img()
>>> TFR.recaptchas[1].click()
>>> TFR.recaptchas[0].click()
>>> TFR.run_inference(TFR.detection_model, TFR.recaptchas[0].img)
['car']
['car']
>>> TFR.run_inference(TFR.detection_model, TFR.recaptchas[2].img)
[]
[]
>>> TFR.run_inference(TFR.detection_model, TFR.recaptchas[5].img)
['car']
['car']
>>> TFR.run_inference(TFR.detection_model, TFR.recaptchas[4].img)
[]
[]
>>> TFR.run_inference(TFR.detection_model, TFR.recaptchas[7].img)
['traffic light']
['traffic light']
>>> 
```