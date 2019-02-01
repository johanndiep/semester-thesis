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

* `dataset.py`: This class is responsible for reading out the information from the rendered dataset. Additionally, it also contains the methods to return the scaled as well as the perturbed information.
* `framework.py`: This class sets up the framework for the optimization process. It initializes the pose with a disturbance which is subsequently mapped to se(3)-formulation. Further, the objective function is defined here.
* `imagegeneration.py`: This class contains methods to generate rendered depth maps, sharp and blurred images at arbitrary poses.
* `meshgeneration.py`: This class contains the construction of the 3D polygon-mesh.
* `optimization.py`: The optimization with PyTorch automatic differentiation package is contained here.
* `posetransformation.py`: Contains the exponential and logarithmic mapping methods.
* `randomizer.py`: Responsible for creating a cardinal directed vector with a defined length as well as an angle axis vector corresponding to an elemental rotation with a defined angle.
* `renderer.py`: Initializes the external renderer module, which is used for generating depth maps.

The [main](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/neural_mesh_renderer/OBMBACPE) folder contains the executables of the classes mentioned above which can be run individually depending on the desired output:

* `pose estimation.py`: This is the main executable file of this project. It finds the pose of the blurred input image via an optimization process.
* `3d representation.py`: This program produces a 3D pointcloud and polygon-mesh representation of the environment in the intial frame. The generated txt- and obj-file can be observed in Meshlab.
* `image generator.py`: This program generates sharp and blurry images at arbitrary poses.

### Docker

The [docker](https://gitlab.com/jdiep/semester-thesis/tree/3-neural-mesh-reprojection/docker) folder contains a Dockerfile able to run the camera tracker from Aebi, Milano and Schnetzler. The following commands can be used: 


***Accessing root***
```bash
sudo -s
```

***Stop all containers***
```bash
docker stop $(docker ps -a -q)
```

***Delete all containers***
```bash
docker rm $(docker ps -a -q)
```

***Delete all images***
```bash
docker rmi $(docker images -q)
```

***Building image***
```bash
docker build -t st-ubuntu .
```

***Run image***
```bash
docker run -it st-ubuntu
```

***Show all images***
```bash
docker images
```

***Show all containers***
```bash
docker ps -a
```

***Copy file to host machine***
```bash
docker cp container_name:/home/semester-thesis/file /home/johann
```

***After running bash***
```bash
apt-get update
apt-get install libmrpt-dev
git clone https://gitlab.com/jdiep/semester-thesis -b 2-dockerfile-to-install-all-dependencies-for-the-project
```

***Run multiple terminal of same container***
```bash
docker exec -it container_name bash
```

## Version

* 2. February 2019: Version 1.0 Release

## Authors

* Johann Diep (MSc Student Mechanical Engineering, jdiep@student.ethz.ch)
* Peidong Liu (Supervisor, peidong.liu@inf.ethz.ch)