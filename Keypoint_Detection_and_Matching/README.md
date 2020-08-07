# Simple KeyPoint Tracker

The goal of this exercise is to build a simple Key-point Detector and Tracker. This helps to familiarize with concepts like Feature Detection, Feature Description, Feature Matching and Tracking.



## Feature Detection

Consider the original image:

<img src="./data/000000.png" alt="drawing" >



Applying the corner response function on the above image,

<img src="./outputs/harris_response.png" alt="drawing" >

Selecting the Keypoints (corners) using the above responses,

<img src="./outputs/harris_keypoints.png" alt="drawing" >



## Feature Description

16 Patch-Based Descriptors with the highest response scores,

<img src="./outputs/descriptors.png" alt="drawing" >



## Feature Matching and Tracking

Feature Matching between two consecutive frames,

<img src="./outputs/matching_output.png" alt="drawing" >



## KeyPoint Detection and Tracking



[![KeyPoint Detection and Tracking](http://img.youtube.com/vi/GeoWAAAaFc0/0.jpg)](http://www.youtube.com/watch?v=GeoWAAAaFc0 "KeyPoint Detection and Tracking")



