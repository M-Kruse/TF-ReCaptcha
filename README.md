# TF-ReCaptcha
Tensorflow project to experiment with defeating ReCaptcha

This is a project to solve the ReCaptcha puzzle using Selenium and Tensorflow or other object detection frameworks.

I noticed a couple things

* You can sometimes force recaptcha to fall back to the easiest type, the fire hydrant, if you repeatedly fail the other tests.
* Sometimes the images look like they have been fuzzed to counter NN object detection. I've heard of techniques to fool object detection while still making the image human readable and I think this is being used here

You can interact with individual ReCaptcha puzzle elements through the RecaptchaElement class. Currently this supports clicking the element and rendering the image of the element.