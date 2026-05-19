<h1 align=center font-weight:100> <strong><i>MBA-SLAM:</i></strong> Motion Blur Aware Dense Visual SLAM with Radiance Fields Representation</h1>

<h3 align="center">TPAMI 2025</h3>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ArXiv-2411.08279-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.08279)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://wangpeng000.github.io/MBA-SLAM/)

</h5>

## Overview
<p align="center">
    <img src="./assets/MBA-SLAM_compressed.gif" alt="Overview" style="width:75%; height:auto;">
</p>

<div>
Given a sequence of severe motion blurred images and depth, MBA-SLAM can accurately estimate the <strong><i>local camera motion trajectory</i></strong> of each blurred image within the exposure time and recovers the high quality 3D scene.
</div>

## Pipeline
<p align="center">
    <img src="./assets/pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
<strong><i>Tracking:</i></strong> Our motion blur-aware tracker directly estimates the camera motion trajectory during the exposure time. <strong><i>Mapping:</i></strong> Our mapper generates virtual sharp images along the camera trajectory.
</div>

## Quickstart

### 1. Setup environment (Both setup options are still under testing and will be updated continuously.)

MBA-SLAM requires an NVIDIA GPU and CUDA. We provide two ways to set up the environment.

#### Option A: Conda environment

Create the Python environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate eslam
pip install -r requirements.txt
pip install pypose --no-deps
```

Build and install the MBA-VO extension:

```bash
cd external/MBA-VO
cmake -GNinja -B build
cmake --build build -t install
cd python
pip install .
cd ../../..
```

If `cmake --build build -t install` cannot write to `/usr/local`, run that install step with `sudo`, then run `sudo ldconfig`. If CMake cannot find Eigen, OpenCV, or CUDA, install the corresponding system packages first, or use the Docker environment below.

#### Option B: Docker environment

The project Docker setup uses two images. `Dockerfile.dev` builds the base environment and tags it as `eslam-base`; `Dockerfile` then uses `FROM eslam-base` and installs MBA-VO plus the remaining Python requirements.

Build the base image:

```bash
docker build -f Dockerfile.dev -t eslam-base .
```

Build the MBA-SLAM image:

```bash
docker build -f Dockerfile -t mba-slam .
```

Run the container from the project root:

```bash
docker run --gpus all --ipc=host -it --rm \
  -v "$(pwd)":/workspace/MBA-SLAM \
  -w /workspace/MBA-SLAM \
  mba-slam bash
```

Inside the container, you can quickly check CUDA and MBA-VO:

```bash
python -c "import torch; import pymbavo; print('cuda:', torch.cuda.is_available())"
```

The Docker build compiles several CUDA/C++ dependencies and can take a long time. Make sure the host has Docker, the NVIDIA driver, and NVIDIA Container Toolkit installed.

### 2. Prepare data and config

Put the dataset under the path used by the config file. For example, `configs/MbaVo/archviz1.yaml` expects:

```yaml
data:
  input_folder: Datasets/MBA-VO/archviz1
  output: output/MBA-VO/archviz1
```

### 3. Training

Run MBA-SLAM on the ArchViz1 example config:

```bash
python run.py ./configs/MbaVo/archviz1.yaml
```

Outputs are written to the config's `data.output` directory.

### 4. Evaluate ATE

After training, evaluate the absolute trajectory error with:

```bash
python ./src/tools/eval_ate.py ./configs/MbaVo/archviz1.yaml
```

### 5. Evaluate reconstruction

Evaluate 3D and 2D reconstruction metrics with:

```bash
python ./src/tools/eval_recon.py --rec_mesh <your.ply path> --gt_mesh <gt.ply path> -3d -2d
```

## Citation

If you find this useful, please consider citing our paper:

```bibtex
@article{wang2025mbaslam,
      title     = {MBA-SLAM: MBA-SLAM: Motion Blur Aware Dense Visual SLAM with Radiance Fields Representation},
      author    = {Wang, Peng and Zhao, Lingzhe and Zhang, Yin and Zhao, Shiyu and Liu, Peidong},
      journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year      = {2025},
      publisher = {IEEE}
  }

  or

@article{wang2024mbaslam,
      title     = {MBA-SLAM: Motion Blur Aware Gaussian Splatting SLAM},
      author    = {Wang, Peng and Zhao, Lingzhe and Zhang, Yin and Zhao, Shiyu and Liu, Peidong},
      journal   = {arXiv preprint arXiv:2411.08279},
      year      = {2024}
  }
```

## Acknowledgment

We would like to thank the authors of [CoSLAM](https://github.com/danping/CoSLAM), [ESLAM](https://github.com/idiap/ESLAM), [gsplat](https://github.com/nerfstudio-project/gsplat), and [MonoGS](https://github.com/muskie82/MonoGS) for their excellent open-source projects and helpful discussions.
