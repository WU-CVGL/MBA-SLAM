FROM nvcr.io/nvidia/pytorch:22.12-py3
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0 TORCH_CUDA_ARCH_LIST="8.6;8.9" MAX_JOBS=32 CMAKE_VERSION=3.21.0
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        autoconf automake autotools-dev build-essential ca-certificates software-properties-common \
        make cmake ninja-build yasm pkg-config checkinstall ccache doxygen gdb \
        gcc gcc-multilib g++ g++-multilib gfortran \
        git nano curl wget unzip p7zip-full xz-utils zstd \
        daemontools net-tools cloc htop iftop iotop rsync \
        libatlas-base-dev libboost-all-dev libcereal-dev libcgal-dev \
        libcxsparse3 libeigen3-dev liblapack-dev liblapacke-dev \
        metis libmetis-dev libopenblas-dev libsuitesparse-dev \
        libgtk2.0-dev libgtk-3-dev libglew-dev libgl1-mesa-dev libgl1-mesa-glx mesa-utils \
        libgflags-dev libgoogle-glog-dev libgtest-dev libprotobuf-dev protobuf-compiler \
        libgphoto2-dev libhdf5-dev libjpeg-dev libpng-dev libtiff-dev libtiff-dev \
        libavcodec-dev libavformat-dev libavutil-dev libavdevice-dev libavresample-dev \
        libpostproc-dev libpq-dev libswscale-dev \
        libtbb2 libtbb-dev libxine2-dev zlib1g-dev \
        krb5-user libkrb5-dev libssl-dev libtool libyaml-dev libyaml-cpp-dev \
        ibverbs-providers libibverbs1 librdmacm1 \
        libnuma1 libnuma-dev libpmi2-0-dev \
        libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
        libfaac-dev libmp3lame-dev libvorbis-dev \
        libopencore-amrnb-dev libopencore-amrwb-dev \
        libv4l-dev v4l-utils \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        openssh-server openssh-client nfs-common\
        colmap ffmpeg \
    && apt-get purge -y unattended-upgrades \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TBB & Determined AI & pip stuff && torchvision
ARG TBB_BASEURL=https://github.com/oneapi-src/oneTBB/releases/download
ARG TBB_VERSION=4.4.5
ARG TBB_DIR=tbb44_20160526oss
ARG TBB_SAVEPATH="/tmp/tbb.tgz"
ARG TBB_LIB_DIR="intel64/gcc4.4"
ARG TBBROOT=/tmp/$TBB_DIR
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config JUPYTER_DATA_DIR=/run/determined/jupyter/data JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
COPY dockerfile_scripts /tmp/det_dockerfile_scripts
COPY requirements.txt /tmp
RUN wget "${TBB_BASEURL}/${TBB_VERSION}/${TBB_DIR}_lin.tgz" -O $TBB_SAVEPATH &&\
    tar -C /tmp -xf "${TBB_SAVEPATH}" &&\
    cp -R $TBBROOT/lib/$TBB_LIB_DIR/* /usr/local/lib/ &&\
    cp -R $TBBROOT/include/ /usr/local/include/ &&\
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install -U pip &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r /tmp/det_dockerfile_scripts/notebook-requirements.txt &&\
    pip install cmake==${CMAKE_VERSION} &&\
    pip install taichi-nightly -i https://pypi.taichi.graphics/simple/ &&\
    pip install -r /tmp/requirements.txt &&\
    /tmp/det_dockerfile_scripts/add_det_nobody_user.sh &&\
    /tmp/det_dockerfile_scripts/install_libnss_determined.sh &&\
    cd /tmp &&\
    git clone https://github.com/pytorch/vision.git &&\
    cd vision &&\
    git checkout tags/v0.15.1 &&\
    python setup.py install &&\
    cd /tmp &&\
    git clone https://github.com/rusty1s/pytorch_scatter &&\
    cd pytorch_scatter &&\
    git checkout tags/2.1.1 && python setup.py install &&\
    rm -rf /tmp/*

# Install Vulkan
# RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - &&\
#     wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.239-focal.list https://packages.lunarg.com/vulkan/1.3.239/lunarg-vulkan-1.3.239-focal.list &&\
#     apt-get update &&\
#     apt-get install -y vulkan-sdk &&\
#     apt-get clean

# Build NeRF-SLAM
WORKDIR /workspace
RUN git clone https://github.com/DmitriZhao/NeRF-SLAM --recursive &&\
    cd NeRF-SLAM &&\
    cmake ./thirdparty/instant-ngp -B build_ngp &&\
    cmake --build build_ngp --config RelWithDebInfo -j &&\
    cmake ./thirdparty/gtsam -DGTSAM_BUILD_PYTHON=1 -B build_gtsam &&\
    cmake --build build_gtsam --config RelWithDebInfo -j -t install python-install python-test &&\
    python setup.py install
