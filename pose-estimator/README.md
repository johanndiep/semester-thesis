pose-estimator
==============

This repo contains all code for our motion blur aware camera pose tracker.
This is a project for the 3D vision seminar at ETH Zürich.

This is the work of a semester thesis based on the project done by the 3D Vision group.


Docker commands 
------------

### Accessing root

```bash
sudo -s
```

### Stop all containers

```bash
docker stop $(docker ps -a -q)
```

### Delete all containers

```bash
docker rm $(docker ps -a -q)
```

### Delete all images

```bash
docker rmi $(docker images -q)
```

###building image

```bash
docker build -t st-ubuntu .
```

###run image

```bash
docker run -it st-ubuntu
```

Installation
------------

### Install Dependencies

- [cmake] (min 3.10)
- ceres-solver 1.13 (refer to official [ceres docs])
- openCV 3 (refer to official docs for your platform e.g [openCV linux])
- mrpt 1.5 (refer to [mrpt])
- Eigen 3 (refer to [eigen])

#### Note on macOS

Some packages can be installed using [homebrew].

```bash
brew install ceres-solver
brew install opencv3 --with-contrib
```

Project Structure
-----------------

There are 3 top level folders.

The `lib` folder contains the main part of our code. It contains the three main steps (reprojection, blurring, solving).
All important functions should be declared in header files which then can be used in `cmd` executables.
The `cmd` folder contains the executables which will make the library code runnable. It will make use of the libraries in `lib`.

Finally `test` contains some unit tests for our library.

For further information of binaries refer to the header comment in the `cmd` folder.

Geometry
--------

For geometry and matrix operations we use [eigen] and [mrpt] which has nice pose
(combination of position and orientation) objects. Those libraries contain functions
to convert between different representations of orientations (e.g. quaternions,
rotation matrices and angular vectors).

Datasets
--------

For evaulation you may want to use the following datasets provided by Peidong Liu:
* [urban-city-dataset](https://drive.google.com/open?id=1sIfWpYLCarmWdwJlvvlIy_wzFzOG5g0H)
* [realistic-dataset](https://drive.google.com/open?id=1HzoGurrq1VxxNmCyhBlb2L49sn3i1JiW)

Authors
-------


* Tim Aebi (tiaebi@student.ethz.ch)
* Francesco Milano (fmilano@student.ethz.ch)
* Christoph Schnetzler (cschnetz@student.ethz.ch)

* Johann Diep (jdiep@student.ethz.ch) - ST student

Supervisors:
* Peidong Liu
* Vagia Tsiminaki


[ceres docs]: http://ceres-solver.org/installation.html
[openCV linux]: https://docs.opencv.org/3.1.0/d7/d9f/tutorial_linux_install.html
[cmake]: https://cmake.org/install/
[homebrew]: https://brew.sh
[eigen]: http://eigen.tuxfamily.org
[mrpt]: https://www.mrpt.org