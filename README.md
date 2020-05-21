# TF-ReCaptcha

This is a project to solve the ReCaptcha puzzle using Selenium and Tensorflow Object Detection API or maybe another object detection framework.

You can interact with individual ReCaptcha puzzle elements through the RecaptchaElement class. Currently this supports clicking the element and rendering the image of the element.

I noticed a couple things about recaptcha so far:

* You can sometimes force recaptcha to fall back to the easiest type, the fire hydrant, if you repeatedly fail the other tests.
* Sometimes the images look like they have been fuzzed to counter NN object detection. I've heard of techniques to fool object detection while still making the image human readable and I think this is being used here

Farming mode won't attempt to solve it, it will just collect the image(s) and fail it intentionally. The point is to collect as many images as possible for additional training.