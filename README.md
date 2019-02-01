# Optimization-based Motion Blur aware Camera Pose Estimation

This GitLab repository contains the Python code for the optimization-based motion blur aware camera pose tracker.

## Context

Many current robotics applications, especially autonomous mobile robots, require a robust position and orientation estimation in order to perform autonomous or semi-autonomous tasks. Vision-based techniques such as visual odometry (VO) or simultaneous location and mapping (SLAM) are used to estimate camera movements from a sequence of images. For large scale deployment as the fundamental module for pose estimation, these methods need to work robustly under any circumstances. Unfortunately, they come with practical requirements. In order to extract apparent motions, some conditions must be fulfilled, such as the presence of sufficient illuminations and textures in the environment, large enough scene overlaps between consecutive frames or the dominance of static scenes. The work in the context of this semester thesis is addressing the latter case, in particular the occurrence of motion blur due to rapid camera movements in combination with a long exposure time.


### Installation

This installation was tested for Ubuntu 16.04 LTS. For other operating sys-
tems, changes or additional packages might be required. The following
packages were used:

* [Python 3.6.5](https://www.python.org/downloads/source/)
* [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive/)
* [PyTorch](https://pytorch.org/)
* [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) (A modified version was used for this project which is not publicly available at the time of writing.)
* [pyquaternion](http://kieranwynn.github.io/pyquaternion/)
* [meshzoo 0.4.3](https://pypi.org/project/meshzoo/)
* [OpenCV 3.4.4](https://www.learnopencv.com/install-opencv-3-4-4-on-ubuntu-16-04/)
* [tqdm](https://github.com/tqdm/tqdm)
* [scikit-image 0.13.1](http://scikit-image.org/docs/0.13.x/install.html)
* [matplotlib 3.0.2](https://matplotlib.org/users/installing.html) (optional)
* [MeshLab 2016](http://www.meshlab.net/#download/) (optional)

### Dataset

For evaluation, the rendered dataset by Peidong Liu was used:

* [Realistic Rendering](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/RelisticRendering-dataset)
* [Urban City](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/UrbanCity%20dataset)

### Code Structure

The [source code](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/neural_mesh_renderer/OBMBACPE/lib) folder contains all the essential classes with its correspond-
ing methods:
* `dataset.py`


## Version

* 2. February 2019: Version 1.0 Release

## Authors

* Johann Diep (MSc Student Mechanical Engineering, jdiep@student.ethz.ch)
* Peidong Liu (Supervisor, peidong.liu@inf.ethz.ch)