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
  htop nano wget git unzip


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

## Cleanup
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
  rm -rf /opt/*
