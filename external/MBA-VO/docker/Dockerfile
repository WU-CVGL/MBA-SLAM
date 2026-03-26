FROM harbor.cvgl.lab/nvidia/pytorch:22.12-py3
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    rm -rf /opt/hpcx/ &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Determined requirements and common tools
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        ccache doxygen graphviz plantuml \
        daemontools krb5-user ibverbs-providers libibverbs1 \
        libkrb5-dev librdmacm1 libssl-dev libtool \
        libnuma1 libnuma-dev libpmi2-0-dev \
        openssh-server openssh-client pkg-config nfs-common \
        ## Tools
        git curl wget unzip nano vim-tiny net-tools sudo htop iotop \
        cloc rsync screen tmux xz-utils software-properties-common \
        ## Deps
        ffmpeg \
        libatlas-base-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-test-dev \
        libcereal-dev \
        libcgal-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libgflags-dev \
        libglew-dev \
        libgoogle-glog-dev \
        libhdf5-dev \
        libmetis-dev \
        libprotobuf-dev \
        libqt5opengl5-dev \
        libsqlite3-dev \
        libsuitesparse-dev \
        protobuf-compiler \
        python3-pip \
        qtbase5-dev \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Conda and Determined AI stuff
WORKDIR /tmp
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.1 &&\
    pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    pip install -r additional-requirements.txt &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*

# Install libfmt>7.0 (or Sophus will use a crazy fmt macro)
ARG FMT_VERSION=8.1.1
RUN cd /tmp &&\
    wget https://github.com/fmtlib/fmt/releases/download/${FMT_VERSION}/fmt-${FMT_VERSION}.zip &&\
    unzip fmt-${FMT_VERSION}.zip &&\
    cd fmt-${FMT_VERSION} &&\
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=RELEASE &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*

# Install Sophus
RUN cd /tmp &&\
    git clone https://github.com/strasdat/Sophus &&\
    cd Sophus &&\
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=RELEASE &&\
    cmake --build build -t install &&\
    ldconfig && rm -rf /tmp/*
