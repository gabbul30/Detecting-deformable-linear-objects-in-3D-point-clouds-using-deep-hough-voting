ARG CUDA="10.0"
ARG UBUNTU="18.04"
ARG ARCH=""

#FROM nvidia/cuda${ARCH}:${CUDA}-devel-ubuntu${UBUNTU}
#from nvidia/cuda:10.0-devel-ubuntu18.04
from ian40306/cuda10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="AMM Lab"

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
LABEL maintainer="Yuxuan Yang"


# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub &&\
    apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl ssh vim unzip htop openssh-client wget libgtk2.0-dev\
    build-essential cmake libopenblas-dev 

ENV CUDNN_VERSION 7.6.5.32
#RUN apt-get install -y --no-install-recommends \ 
#    libcudnn7=$CUDNN_VERSION-1+cuda10.0 \    
#    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
#    apt-mark hold libcudnn7
    

# Python 3.7
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems

RUN apt-get update && apt-get install -y  python3.6 python3.6-dev python3-pip python3-tk \
   python-opengl python3-opengl&& \
   pip3 install --upgrade pip setuptools wheel

# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy==1.19.5 scipy scikit-learn scikit-image imgaug  IPython[all] matplotlib==3.3.4 opencv-python==4.5.3.56\
    Cython requests PyYAML h5py tqdm Pillow==6.2.1 progressbar2 \
    jupyter_contrib_nbextensions notebook==6.4.10 \
    && jupyter  contrib nbextension install

# Install pytorch  
#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 kornia tensorboard 
RUN pip3 install torch==1.2.0 torchvision==0.4.0

RUN pip3 install  arrow termcolor  igraph plyfile trimesh==2.35.39 networkx==2.2 geomdl tensorflow==1.14 tensorboard
RUN python3 -m pip install --no-cache-dir --upgrade open3d
#ENV CUDNN_VERSION 7.6.5.32
#RUN apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda10.0 && \
#    apt-mark hold libcudnn7 && \
#    rm -rf /var/lib/apt/lists/*


#COPY ./agx.lic /opt/Algoryx/AgX-2.29.4.0/
RUN apt-get install --no-install-recommends -y libegl1 libgl1 libgomp1 gedit libdrm2 libedit2 libexpat1 libglapi-mesa libllvm10 libx11-xcb1 \
    libxcb-dri2-0 libxcb-glx0 libxcb-shm0 libxcb-xfixes0 libxfixes3 \
    libxxf86vm1 mesa-utils libpcl-dev pcl-tools
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888
# VNC Server
EXPOSE 5900

#ENTRYPOINT ["tail", "-f", "/dev/null"]
##CMD ["/bin/sh"]
