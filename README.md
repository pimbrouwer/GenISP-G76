# GenISP-G76
**Paper reproducibility project**  
The implementation shown in this GitHub is from scratch.  
## Introduction
Object detection is a well known task in the practice of computer vision. A lot of off-the-self object detectors exist nowadays with amazingly high accuracy. However, object detection in low-light conditions still remains problematic. 

This blog post tries to reproduce a paper that tries to solve this exact problem, detecting objects in low-light conditions. The authors Igor Morawski, Yu-An Chen, Yu-Sheng Lin, Shusil Dangi, Kai He, and Winston H. Hsu proposed a neural workflow in their paper “GenISP: Generalizing Image Signal Processing for Low-Light Object Detection”. Their goal is to find 12 parameters for the white balancing and colour correction of the image so that the object detector can have better results on dark images.

In this report, we try to reproduce their workflow by using the method mentioned in their paper, using the same dataset and thus trying to ensure the same result. Furthermore we will discuss the ease at which this paper was to reproduce.

## Method
Firstly, we downloaded the Nikon dataset which contains around 4000 images and the ground truth bounding box annotations of people, bicycles and cars. These images are taken in different low-light circumstances, differing between street lights and full dark pictures.

The paper starts with preprocessing dark images shot on a Nikon camera. These images are saved in the RAW format. This format contains the image data as intensity for every pixel and every colour channel. The main advantage of working with RAW files is that there is no preprocessing done yet by the camera, therefore these files are independent of camera type which results in more consistent results when used for object detection, even when different cameras are used. Each pixel contains a 2-by-2 array for different colour channels in the RGrGbB format. The colour channels are then packed so a single image now consists of 4 different channels so every channel is now its own array. The two green channels are averaged so the image is now in standard RGB format. 

