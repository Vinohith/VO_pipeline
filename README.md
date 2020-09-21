# VO_pipeline



## Requirements

* Create the environment from the `environment.yml` file:

```
conda env create -f environment.yml
```

* Additionally we will also require:
  * [g2o](https://github.com/uoip/g2opy) for optimization
  * [pangolin](https://github.com/uoip/pangolin) for visualization



## Folder Structure

* [Keypoint_Detection_and_Matching](https://github.com/Vinohith/VO_pipeline/tree/master/Keypoint_Detection_and_Matching) : This folder contains code for a simple harris corner detection and matching script from scratch.
* [VO](https://github.com/Vinohith/VO_pipeline/tree/master/VO) : This folder has the code for the visual odometry pipeline including g2o optimization.
* [pangolin_example](https://github.com/Vinohith/VO_pipeline/tree/master/pangolin_example) : This folder has the code for visualization of a trajectory using pangolin.