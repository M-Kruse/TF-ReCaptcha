# TF-ReCaptcha

This is a project to solve the ReCaptcha puzzle using Selenium and Tensorflow Object Detection API or maybe another object detection framework.

You can interact with individual ReCaptcha puzzle elements through the RecaptchaElement class. Currently this supports clicking the element and rendering the image of the element.

The 3x3 seems easier to solve, so that is the first goal.

I noticed some things about recaptcha so far:

* You can sometimes force recaptcha to fall back to the easiest type, the fire hydrant, if you repeatedly fail the other tests.
* Sometimes the images look like they have been fuzzed to counter NN object detection. I've heard of techniques to fool object detection while still making the image identifiable by a human. I think this is being used here and it makes sense.
* The number of anti-NN object detection images seems to increase the more you access the recaptcha from a single IP.
* You can access the 3x3 as a single image, otherwise the grid elements have single image sources.
* The 4x4 is a larger single image also with a grid overlay
* I haven't seen any 4x4 puzzle type where the image looks like its been modified against object detection. I'm wondering if that is because building a system for the 4x4 is much more difficult.

Harvesting mode won't attempt to solve it, it will just collect the image(s) and fail it intentionally. The point is to collect as many images as possible for training.

Here is an example usage of interacting with the puzzle elements, it is not trained for crosswalks but you should get the idea.

https://streamable.com/8rvkr4

```
Python 3.8.2 (default, Apr 27 2020, 15:53:34) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from tf_recaptcha import TFRecaptcha
>>> T = TFRecaptcha()
[INFO] Skipping image class directory creation for fire hydrant
[INFO] Skipping image class directory creation for fire hydrants
[INFO] Skipping image class directory creation for bicycle
[INFO] Skipping image class directory creation for bicycles
[INFO] Skipping image class directory creation for traffic light
[INFO] Skipping image class directory creation for traffic lights
[INFO] Skipping image class directory creation for bus
[INFO] Skipping image class directory creation for buses
[INFO] Skipping image class directory creation for taxis
[INFO] Skipping image class directory creation for cars
[INFO] Skipping image class directory creation for motorcycle
[INFO] Skipping image class directory creation for motorcycles
[INFO] Skipping image class directory creation for crosswalk
[INFO] Skipping image class directory creation for crosswalks
[INFO] Skipping image class directory creation for unknown
>>> T.init_browser()
True
>>> T.init_tf()
>>> T.open_test_recaptcha()
[INFO] Attempting to bypass 4x4 puzzle type
>>> T.generate_recaptcha_classes()
>>> T.recaptchas
[<tf_recaptcha.RecaptchaElement object at 0x7ff9c84ab400>, <tf_recaptcha.RecaptchaElement object at 0x7ff9c84aba30>, <tf_recaptcha.RecaptchaElement object at 0x7ff9c84ab2b0>, <tf_recaptcha.RecaptchaElement object at 0x7ff9c84ab910>, <tf_recaptcha.RecaptchaElement object at 0x7ffa58cfc820>, <tf_recaptcha.RecaptchaElement object at 0x7ff9c849e190>, <tf_recaptcha.RecaptchaElement object at 0x7ffa58d1ef70>, <tf_recaptcha.RecaptchaElement object at 0x7ff9c849e100>, <tf_recaptcha.RecaptchaElement object at 0x7ff9c849e2b0>]
>>> for r in T.recaptchas:
...  T.run_inference(r.img)
... 
['car']
['car']
[]
['traffic light', 'suitcase', 'potted plant']
['bird', 'car', 'truck']
['bicycle']
['bicycle']
['bicycle', 'chair', 'car']
['person', 'truck']
>>> for r in T.recaptchas:
...  if T.T.run_inference(r.img)


```