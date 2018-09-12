##Installation of all the dependencies according to README from 3D Vision project

## Install the OS image
FROM ubuntu:xenial

## Author
MAINTAINER Johann Diep <jdiep@student.ethz.ch>

## Pre-installation Actions, Display the kernel version
RUN uname -m && cat /etc/*release
RUN uname -r

## Initial update and install of dependencies
RUN apt-get -y update && apt-get install -y software-properties-common python-software-properties

## Installation of all dependencies
RUN apt-get update && apt-get install -y \
  libgtk2.0-dev \
libjpeg-dev \
  libjasper-dev \
  libopenexr-dev cmake python-dev \
  python-numpy python-tk libtbb-dev \
  libeigen2-dev yasm libfaac-dev \
  libopencore-amrnb-dev libopencore-amrwb-dev \
  libtheora-dev libvorbis-dev libxvidcore-dev \
  libx264-dev libqt4-dev libqt4-opengl-dev \
  sphinx-common libv4l-dev libdc1394-22-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libglew-dev libboost-all-dev \
  htop nano wget git unzip \
  libgoogle-glog-dev libatlas-base-dev libeigen3-dev \
  libsuitesparse-dev 


## Install cmake 3.12.2
WORKDIR /opt
RUN wget https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh
RUN chmod +x /opt/cmake-3.12.2-Linux-x86_64.sh
RUN mkdir -p /opt/cmake
RUN sh /opt/cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

## Test compiler
RUN gcc --version
RUN g++ --version
RUN cmake --version

## Install Ceres Solver
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cmake ceres-solver
RUN make -j3
RUN make test
RUN make install

## Install OpenCV 3.0
RUN apt-get update && apt-get install -y --force-yes build-essential cmake git libgtk2.0-dev \
  pkg-config libavcodec-dev libavformat-dev libswscale-dev \
  libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
  python3-dev python3-tk python3-numpy && apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir ~/opencv

RUN cd ~/opencv && git clone https://github.com/Itseez/opencv_contrib.git && cd opencv_contrib && git checkout 3.0.0
RUN cd ~/opencv && git clone https://github.com/Itseez/opencv.git && cd opencv && git checkout 3.0.0

RUN cd ~/opencv/opencv && mkdir release && cd release && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D INSTALL_C_EXAMPLES=ON \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D BUILD_EXAMPLES=ON \
  -D WITH_OPENGL=ON \
  -D WITH_V4L=ON \
  -D WITH_XINE=ON \
  -D WITH_TBB=ON ..

RUN cd ~/opencv/opencv/release && make -j $(nproc) && make install

## Source code for motion blur algorithm
RUN git clone https://gitlab.com/jdiep/semester-thesis.git /home/motion-blur-cam-pose-tracker
