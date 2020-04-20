FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

## Base packages for ubuntu
RUN apt-get clean && \
    apt-get update -qq && \
    apt-get install -y \
        sudo \
        gosu \
        git \
        wget \
        bzip2 \
        htop \
        nano \
        g++ \
        gcc \
        make \
        build-essential \
        software-properties-common \
        apt-transport-https \
        libhdf5-dev \
        libgl1-mesa-glx \
        openmpi-bin \
        graphviz

## Download and install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

# Install python and upgrade pip version
RUN conda install -y python=3.6 && \
    pip install --upgrade pip

## Setup order is important (GDAL, Rasterio, OpenCV)! - otherwise it won't import due to dependency conflict
RUN add-apt-repository ppa:ubuntugis/ppa \
 && apt-get update \
 && apt-get install -y python-numpy gdal-bin libgdal-dev \
 && pip install \
    rasterio==1.0b2 \
    opencv-python==3.4.1.15

### Build libspatialindex from source because conda's libspatialindex conflicts with GDAL
RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz -O /tmp/spatialindex-src.tar.gz && \
    tar -xvf /tmp/spatialindex-src.tar.gz -C /tmp
WORKDIR /tmp/spatialindex-src-1.8.5
RUN ./configure && make && make install && ldconfig && pip install Rtree==0.8.3

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME